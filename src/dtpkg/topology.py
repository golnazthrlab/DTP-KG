"""Local graph descriptors with fold-specific edge exclusions.

The transductive study uses the optimized, unweighted radius-two backend with
spectral Katz scaling and no closeness cutoff. Defaults retain the legacy sparse
graph settings; use :func:`dtpkg.topology_config.make_extractor` for the current
study settings. Raw per-drug features and cache files are private run artifacts.
"""

import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla

# Populated in the parent before forking a worker pool; inherited copy-on-write.
_FORK_STATE = {}

# Bump when a change alters computed feature values, so cached tables from an
# earlier definition are never reused.
FEATURE_CODE_VERSION = "2026-09-17.b"


def _worker_chunk(drug_ids):
    """Run the optimized per-drug path inside a forked worker."""
    extractor = _FORK_STATE["extractor"]
    graph = _FORK_STATE["graph"]
    context = _FORK_STATE["context"]
    alpha = _FORK_STATE["alpha"]
    results = []
    for drug in drug_ids:
        started = time.perf_counter()
        try:
            feats = extractor._compute_local_optimized(graph, drug, context, alpha)
            results.append((drug, "computed", feats, dict(extractor._local_status),
                            time.perf_counter() - started, None))
        except Exception as exc:  # reported per drug, never silently dropped
            results.append((drug, "error", None, None,
                            time.perf_counter() - started, repr(exc)))
    return results


class TopoFeatExtractor:
    def __init__(self,
                 graph_path: str = None,
                 use_betweenness: bool = False,
                 use_protein_neighborhood: bool = False,
                 protein_feat_path: str = None,
                 use_weighted: bool = False,
                 backend: str = "reference",
                 graph: "nx.Graph" = None,
                 closeness_max_nodes: int = 200,
                 katz_alpha_mode: str = "fixed",
                 katz_alpha_safety: float = 0.5,
                 katz_max_iter: int = 100,
                 katz_tol: float = 1e-3,
                 n_jobs: int = 1,
                 cache_dir=None,
                 ):
        """
        closeness_max_nodes
            Ego sizes at or above this return close_local=0. 200 reproduces the
            legacy sparse-graph runs. It is a reference-backend cost shortcut, not a
            definition: on a DDI-containing graph nearly every ego exceeds it, so
            close_local collapses to a constant. Pass None to compute it for every
            ego (free under `optimized`, expensive under `reference`).
        katz_alpha_mode
            "fixed" uses the alpha passed to the compute call (0.005 in the
            legacy sparse-graph runs). "spectral" instead uses katz_alpha_safety/lambda_max
            of each ego, which is the condition under which the Katz series
            converges at all. On DDI-dense egos alpha=0.005 gives
            alpha*lambda_max > 1, so the series diverges and every value falls back
            to 0. Switching mode changes the descriptor and must be reported.
        n_jobs
            Drug-level process parallelism for the optimized backend. Workers are
            forked after the shared sparse state is built, so the graph is not
            pickled. Ignored by the reference backend and by n_jobs=1.
        cache_dir
            Optional directory for raw feature tables, keyed by graph content,
            mask, requested drugs and every feature setting below. Nothing is
            reused across a different mask, graph or setting.
        """

        if backend not in {"reference", "optimized"}:
            raise ValueError("backend must be 'reference' or 'optimized'")
        if katz_alpha_mode not in {"fixed", "spectral"}:
            raise ValueError("katz_alpha_mode must be 'fixed' or 'spectral'")
        self.backend = backend
        self.closeness_max_nodes = closeness_max_nodes
        self.katz_alpha_mode = katz_alpha_mode
        self.katz_alpha_safety = float(katz_alpha_safety)
        self.katz_max_iter = int(katz_max_iter)
        self.katz_tol = float(katz_tol)
        self.n_jobs = int(n_jobs)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_diagnostics = pd.DataFrame()
        self.last_setup_seconds = 0.0
        self.last_cache_status = "disabled"
        self._graph_hash = None

        if (graph_path is None) == (graph is None):
            raise ValueError("supply exactly one of graph_path or graph")
        self.G = nx.read_graphml(graph_path) if graph is None else graph
        self.graph_path = str(graph_path) if graph_path is not None else None

        node_types = nx.get_node_attributes(self.G, "type")
        self.drug_nodes = [n for n, t in node_types.items() if t.lower() == "drug"]

        print(f"   Graph loaded: {len(self.G)} nodes, {self.G.number_of_edges()} edges")
        print(f"   Found {len(self.drug_nodes)} drug nodes.\n")

        # Core settings
        self.use_betweenness = use_betweenness
        self.use_protein_neighborhood = use_protein_neighborhood

        # --- Load precomputed drug-level GO features if requested ---
        if use_protein_neighborhood:
            if protein_feat_path is None:
                raise ValueError("protein_feat_path is required for GO-neighborhood features")
            print(f"Loading precomputed drug GO-neighborhood features from {protein_feat_path} ...")
            self.drug_go_df = pd.read_csv(protein_feat_path, index_col=0)
            print(f"Loaded {self.drug_go_df.shape[0]} drugs × {self.drug_go_df.shape[1]} features.\n")

            #  last column is entropy
            self.num_of_protein_feats = self.drug_go_df.shape[1]
        else:
            self.drug_go_df = None
            self.num_of_protein_feats = 0

        self.use_weighted = use_weighted

        # --- Base topological feature count ---
        self.num_of_topo_feats = 13 if use_betweenness else 12

        # --- Total feature size per drug ---
        self.num_of_topo_feats = self.num_of_topo_feats + self.num_of_protein_feats

    # ------------------------------------------------------------------
    #                       PROVENANCE AND CACHING
    # ------------------------------------------------------------------

    def feature_settings(self, radius, katz_alpha):
        """Every setting that can change a computed value. Part of the cache key."""
        return {
            "code_version": FEATURE_CODE_VERSION,
            "backend": self.backend,
            "radius": radius,
            "katz_alpha": katz_alpha,
            "katz_alpha_mode": self.katz_alpha_mode,
            "katz_alpha_safety": self.katz_alpha_safety,
            "katz_max_iter": self.katz_max_iter,
            "katz_tol": self.katz_tol,
            "closeness_max_nodes": self.closeness_max_nodes,
            "use_betweenness": self.use_betweenness,
            "use_weighted": self.use_weighted,
            "use_protein_neighborhood": self.use_protein_neighborhood,
        }

    def _cache_key(self, graph, drug_ids, radius, katz_alpha):
        from dtpkg.network.graph_variants import graph_content_hash
        digest = hashlib.sha256()
        digest.update(graph_content_hash(graph).encode())
        digest.update(b"\x1d")
        digest.update(json.dumps(sorted(map(str, drug_ids))).encode())
        digest.update(b"\x1d")
        digest.update(json.dumps(self.feature_settings(radius, katz_alpha),
                                 sort_keys=True, default=str).encode())
        return digest.hexdigest()

    # ------------------------------------------------------------------
    #                        OPTIMIZED BACKEND
    # ------------------------------------------------------------------

    def _optimized_context(self, G, radius):
        """Build numeric state ONLY for this already-masked graph/extraction call."""
        if (self.backend != "optimized" or self.use_weighted
                or self.use_betweenness or radius != 2 or G.is_directed()
                or G.is_multigraph() or nx.number_of_selfloops(G)):
            return None
        nodes = list(G)
        if not nodes:
            return None
        return {
            "index": {node: i for i, node in enumerate(nodes)},
            "adjacency": nx.to_scipy_sparse_array(
                G, nodelist=nodes, weight=None, dtype=float, format="csr"),
            "degree": dict(G.degree()),
            "types": nx.get_node_attributes(G, "type"),
        }

    def _spectral_alpha(self, adjacency, fallback_alpha):
        """alpha = safety / lambda_max, the range where the Katz series converges.

        Falls back to the max-degree bound (lambda_max <= max degree) if the
        eigensolver fails, which is conservative rather than divergent.
        """
        n = adjacency.shape[0]
        if n < 3 or adjacency.nnz == 0:
            return fallback_alpha, None
        try:
            lam = float(abs(sla.eigsh(adjacency, k=1, which="LM", tol=1e-4,
                                      return_eigenvectors=False)[0]))
        except Exception:
            lam = float(np.asarray(adjacency.sum(axis=1)).max())
        if not np.isfinite(lam) or lam <= 0:
            return fallback_alpha, None
        return self.katz_alpha_safety / lam, lam

    def _sparse_katz(self, adjacency, center, alpha):
        """NetworkX's zero-start, beta=1 power iteration, in sparse arithmetic."""
        x = np.zeros(adjacency.shape[0], dtype=float)
        for _ in range(self.katz_max_iter):
            previous = x
            x = alpha * (adjacency @ previous) + 1.0
            if np.abs(x - previous).sum() < len(x) * self.katz_tol:
                scale = 1.0 / math.hypot(*x)
                return float(x[center] * scale)
        raise nx.PowerIterationFailedConvergence(self.katz_max_iter)

    def _closeness_cutoff_hit(self, ego_size):
        return (self.closeness_max_nodes is not None
                and ego_size >= self.closeness_max_nodes)

    def _compute_local_optimized(self, G, node, context, katz_alpha):
        # Exact identities for a simple, loop-free, unweighted radius-2 ego.
        neighbors = list(G[node])
        ego = {node, *neighbors}
        for neighbor in neighbors:
            ego.update(G[neighbor])
        index = context["index"]
        indices = sorted(index[n] for n in ego)
        adjacency = context["adjacency"][indices, :][:, indices]
        center = int(np.searchsorted(indices, index[node]))
        degree = context["degree"]
        types = context["types"]
        neighbor_degrees = [degree[n] for n in neighbors]
        boundary_edges = sum(degree[n] for n in ego) - adjacency.nnz
        first_shell = len(neighbors)
        second_shell = len(ego) - first_shell - 1
        distance_sum = first_shell + 2 * second_shell
        cutoff_hit = self._closeness_cutoff_hit(len(ego))
        closeness = ((len(ego) - 1) / distance_sum
                     if distance_sum and not cutoff_hit else 0)
        if len(ego) == 1:
            closeness = 0.0
        alpha, lam = (katz_alpha, None)
        if self.katz_alpha_mode == "spectral":
            alpha, lam = self._spectral_alpha(adjacency, katz_alpha)
        self._local_status = {
            "backend": "optimized", "closeness_shortcut": cutoff_hit,
            "katz_status": "converged", "ego_edges": adjacency.nnz // 2,
            "katz_alpha_used": alpha, "ego_lambda_max": lam,
        }
        try:
            katz = self._sparse_katz(adjacency, center, alpha)
        except nx.PowerIterationFailedConvergence:
            katz = 0
            self._local_status["katz_status"] = "nonconvergence_zero"
        return {
            "deg_total": degree[node],
            "deg_drug": sum(types.get(n, "").lower() == "drug" for n in neighbors),
            "deg_prot": sum(types.get(n, "").lower() in {"protein", "target"}
                            for n in neighbors),
            "deg_min": np.min(neighbor_degrees) if neighbor_degrees else 0,
            "deg_max": np.max(neighbor_degrees) if neighbor_degrees else 0,
            "deg_mean": np.mean(neighbor_degrees) if neighbor_degrees else 0,
            "deg_std": np.std(neighbor_degrees) if neighbor_degrees else 0,
            "clust_local": nx.clustering(G, node),
            "boundary_ratio": boundary_edges / (len(ego) + 1e-8),
            "close_local": closeness,
            "katz_local": katz,
            "n_nodes_subgraph": len(ego),
        }

    def compute_local_features(self, G, node, radius=2, katz_alpha=0.005):
        context = self._optimized_context(G, radius)
        if context is not None:
            return self._compute_local_optimized(G, node, context, katz_alpha)
        return self._compute_local_reference(G, node, radius, katz_alpha)

    # ------------------------------------------------------------------
    #                        REFERENCE BACKEND
    # ------------------------------------------------------------------

    def _compute_local_reference(self, G, node, radius=2, katz_alpha=0.005):
        weight_arg = "weight" if self.use_weighted else None

        # --- ego graph ---
        t0 = time.time()
        if self.use_weighted:
            subG = nx.ego_graph(G, node, radius=radius, distance="weight")
        else:
            subG = nx.ego_graph(G, node, radius=radius)
        cutoff_hit = self._closeness_cutoff_hit(len(subG))
        self._local_status = {
            "backend": "reference", "closeness_shortcut": cutoff_hit,
            "katz_status": "converged", "ego_edges": subG.number_of_edges(),
            "katz_alpha_used": katz_alpha, "ego_lambda_max": None,
        }
        # print(f"[{node}] ego_graph: {time.time() - t0:.4f}s")

        node_types = nx.get_node_attributes(subG, "type")

        # --- degree features ---
        t1 = time.time()
        deg_total = subG.degree(node, weight=weight_arg)
        deg_drug = sum(
            subG[node][nbr].get("weight", 1.0) if self.use_weighted else 1
            for nbr in subG.neighbors(node)
            if node_types.get(nbr, "").lower() == "drug"
        )
        deg_prot = sum(
            subG[node][nbr].get("weight", 1.0) if self.use_weighted else 1
            for nbr in subG.neighbors(node)
            if node_types.get(nbr, "").lower() in ["protein", "target"]
        )
        # print(f"[{node}] degree_features: {time.time() - t1:.4f}s")

        # --- local degree profile ---
        t2 = time.time()
        neighbor_degrees = [subG.degree(n, weight=weight_arg) for n in subG.neighbors(node)]
        if neighbor_degrees:
            deg_min = np.min(neighbor_degrees)
            deg_max = np.max(neighbor_degrees)
            deg_mean = np.mean(neighbor_degrees)
            deg_std = np.std(neighbor_degrees)
        else:
            deg_min = deg_max = deg_mean = deg_std = 0
        # print(f"[{node}] degree_profile: {time.time() - t2:.4f}s")

        # --- clustering ---
        t3 = time.time()
        clustering = nx.clustering(subG, nodes=[node], weight=weight_arg)[node]
        # print(f"[{node}] clustering: {time.time() - t3:.4f}s")

        # --- boundary ratio ---
        t4 = time.time()
        ego_nodes = set(subG.nodes())
        if self.use_weighted:
            boundary_edges = sum(
                G[n][nbr].get("weight", 1.0)
                for n in ego_nodes for nbr in G.neighbors(n)
                if nbr not in ego_nodes
            )
        else:
            boundary_edges = sum(
                1 for n in ego_nodes for nbr in G.neighbors(n)
                if nbr not in ego_nodes
            )
        boundary_ratio = boundary_edges / (len(ego_nodes) + 1e-8)
        # print(f"[{node}] boundary_ratio: {time.time() - t4:.4f}s")

        # --- closeness ---
        t5 = time.time()
        if not cutoff_hit:
            close_local = nx.closeness_centrality(subG, distance=weight_arg).get(node, 0)
        else:
            close_local = 0
        # print(f"[{node}] closeness: {time.time() - t5:.4f}s")

        # --- Katz ---
        t6 = time.time()
        alpha = katz_alpha
        if self.katz_alpha_mode == "spectral":
            adjacency = nx.to_scipy_sparse_array(
                subG, weight=weight_arg, dtype=float, format="csr")
            alpha, lam = self._spectral_alpha(adjacency, katz_alpha)
            self._local_status["katz_alpha_used"] = alpha
            self._local_status["ego_lambda_max"] = lam
        try:
            katz_local = nx.katz_centrality(
                subG, alpha=alpha, max_iter=self.katz_max_iter, tol=self.katz_tol,
                weight=weight_arg
            ).get(node, 0)

        except Exception as exc:
            katz_local = 0
            self._local_status["katz_status"] = (
                "nonconvergence_zero" if isinstance(exc, nx.PowerIterationFailedConvergence)
                else "error_zero:" + type(exc).__name__)
        # print(f"[{node}] katz: {time.time() - t6:.4f}s")

        # --- betweenness ---
        betw_local = None
        if self.use_betweenness:
            t7 = time.time()
            try:
                k_sample = min(10, len(subG))
                betw_local = nx.betweenness_centrality(subG, k=k_sample, weight=weight_arg).get(node, 0)
            except Exception:
                betw_local = 0
            # print(f"[{node}] betweenness: {time.time() - t7:.4f}s")

        feats = {
            "deg_total": deg_total,
            "deg_drug": deg_drug,
            "deg_prot": deg_prot,
            "deg_min": deg_min,
            "deg_max": deg_max,
            "deg_mean": deg_mean,
            "deg_std": deg_std,
            "clust_local": clustering,
            "boundary_ratio": boundary_ratio,
            "close_local": close_local,
            "katz_local": katz_local,
            "n_nodes_subgraph": len(subG),
        }

        if self.use_betweenness:
            feats["betw_local"] = betw_local

        return feats

    # ------------------------------------------------------------------
    #                           DRIVER
    # ------------------------------------------------------------------

    EMPTY_COLUMNS = [
        "deg_total", "deg_drug", "deg_prot",
        "deg_min", "deg_max", "deg_mean", "deg_std",
        "clust_local", "boundary_ratio",
        "close_local", "katz_local", "betw_local", "n_nodes_subgraph",
    ]

    def _compute_parallel(self, graph, drug_ids, context, katz_alpha, t0):
        """Fork `n_jobs` workers over the already-built shared sparse state."""
        chunks = [list(c) for c in np.array_split(np.array(drug_ids, dtype=object),
                                                  min(len(drug_ids), self.n_jobs * 4))
                  if len(c)]
        _FORK_STATE.update(extractor=self, graph=graph, context=context, alpha=katz_alpha)
        # No thread-limit juggling: the per-drug path is CSR slicing, a CSR
        # matrix-vector product and ARPACK on the same product, all single
        # threaded in SciPy. There is no dense BLAS call to oversubscribe.
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs,
                                     mp_context=get_context("fork")) as pool:
                collected = []
                for done, chunk in enumerate(pool.map(_worker_chunk, chunks), start=1):
                    collected.extend(chunk)
                    print(f"→ chunk {done}/{len(chunks)} "
                          f"({time.perf_counter() - t0:.1f}s elapsed)", flush=True)
        finally:
            _FORK_STATE.clear()
        return collected

    def _compute_topo_features(self, graph, drug_ids, radius=2, katz_alpha=0.005):
        """Compute local features for a list of drug IDs."""
        t0 = time.perf_counter()
        context = self._optimized_context(graph, radius)
        self.last_setup_seconds = time.perf_counter() - t0

        present = [d for d in drug_ids if d in graph]
        records = {}

        parallel = (context is not None and self.n_jobs > 1 and len(present) > 1)
        if parallel:
            collected = self._compute_parallel(graph, present, context, katz_alpha, t0)
        else:
            collected = []
            for i, d in enumerate(present):
                started = time.perf_counter()
                try:
                    if context is None:
                        feats = self._compute_local_reference(graph, d, radius, katz_alpha)
                    else:
                        feats = self._compute_local_optimized(graph, d, context, katz_alpha)
                    collected.append((d, "computed", feats, dict(self._local_status),
                                      time.perf_counter() - started, None))
                except Exception as e:
                    collected.append((d, "error", None, None,
                                      time.perf_counter() - started, repr(e)))
                if (i + 1) % 500 == 0:
                    print(f"→ Processed {i+1}/{len(present)} drugs "
                          f"({time.perf_counter()-t0:.1f}s elapsed)", flush=True)

        rows = {}
        for drug, status, feats, local, seconds, error in collected:
            if status == "computed":
                rows[drug] = feats
                records[drug] = {"drugbank_id": drug, "status": "computed", **local,
                                 "ego_nodes": feats["n_nodes_subgraph"], "seconds": seconds}
            else:
                print(f"Error computing {drug}: {error}")
                records[drug] = {"drugbank_id": drug, "status": "error",
                                 "error": error, "seconds": seconds}

        # Diagnostics and rows follow the requested order, so a parallel run and a
        # serial run return identical tables.
        diagnostics = [records.get(d, {"drugbank_id": d, "status": "missing_node"})
                       for d in drug_ids]
        self.last_diagnostics = pd.DataFrame(diagnostics)
        if not rows:
            print("No valid topological features computed — returning empty DataFrame.")
            return pd.DataFrame(columns=self.EMPTY_COLUMNS)

        df = pd.DataFrame([{"drugbank_id": d, **rows[d]} for d in drug_ids
                           if d in rows]).set_index("drugbank_id")
        self._report_degenerate_columns(df)
        print(f"Finished {len(df)} drugs in {time.perf_counter()-t0:.1f}s total.")
        return df

    def _report_degenerate_columns(self, df):
        """Say out loud when a descriptor carried no information this call.

        A shortcut or a non-convergent Katz produces a constant column that looks
        like a computed feature. Comparing graph variants without noticing this
        attributes a missing feature to the biology.
        """
        constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        diagnostics = self.last_diagnostics
        if not diagnostics.empty and "katz_status" in diagnostics:
            failed = (diagnostics.katz_status == "nonconvergence_zero").sum()
            if failed:
                print(f"WARNING: Katz did not converge for {failed}/{len(diagnostics)} "
                      f"drugs (value set to 0). alpha*lambda_max >= 1 means the "
                      f"series diverges; more iterations cannot fix it.")
        if not diagnostics.empty and "closeness_shortcut" in diagnostics:
            cut = int(diagnostics.closeness_shortcut.eq(True).sum())
            if cut:
                print(f"WARNING: close_local set to 0 by the "
                      f"closeness_max_nodes={self.closeness_max_nodes} shortcut for "
                      f"{cut}/{len(diagnostics)} drugs.")
        if constant:
            print(f"WARNING: constant (uninformative) feature columns: {constant}")

    # ------------------------------------------------------------------
    #                          PUBLIC API
    # ------------------------------------------------------------------

    def compute_for_fold(self,
                         drug_ids,
                         eval_pairs=None,
                         radius=2,
                         katz_alpha=0.005,
                    ):
        """Compute on a fresh fold graph; callers supply all required exclusions.

        eval_pairs masks positive pair-level targets only. Drug-disjoint folds
        need all incident DDI exclusions, including unsampled positives. No
        feature/context cache is shared across calls or distinct folds; the
        optional on-disk cache is keyed by graph content, mask and settings.
        """
        edges_to_remove = []
        if eval_pairs is not None:
            for _, r in eval_pairs.iterrows():
                if r.get("label", 1) == 1:
                    edges_to_remove.append((str(r["drug1"]), str(r["drug2"])))

        print(f"Removing {len(edges_to_remove)} eval DDI edges for leakage safety.")
        fold_graph = self.G.copy() if edges_to_remove else self.G
        fold_graph.remove_edges_from(edges_to_remove)
        print(f"Computing topological features for {len(drug_ids)} drugs...")
        feats = self._cached_compute(fold_graph, drug_ids, radius, katz_alpha)
        if self.use_protein_neighborhood and self.drug_go_df is not None:
            feats = feats.join(self.drug_go_df, how="left").fillna(0)


        print(f"Computed features for {len(feats)} drugs.")
        return feats

    def _cached_compute(self, graph, drug_ids, radius, katz_alpha):
        """Reuse a raw table only when graph content, drugs and settings match."""
        if self.cache_dir is None:
            self.last_cache_status = "disabled"
            return self._compute_topo_features(graph, drug_ids, radius, katz_alpha)
        key = self._cache_key(graph, drug_ids, radius, katz_alpha)
        # Pickle, not parquet: the project environment has no pyarrow, and the
        # cache is a local, disposable artefact keyed by code version.
        table = self.cache_dir / f"topo_{key}.pkl"
        diag = self.cache_dir / f"diag_{key}.pkl"
        if table.exists():
            self.last_cache_status = "hit"
            print(f"Cache hit {key[:12]} — reusing raw topological features.")
            self.last_diagnostics = (pd.read_pickle(diag) if diag.exists()
                                     else pd.DataFrame())
            return pd.read_pickle(table)
        feats = self._compute_topo_features(graph, drug_ids, radius, katz_alpha)
        feats.to_pickle(table)
        self.last_diagnostics.to_pickle(diag)
        (self.cache_dir / f"key_{key}.json").write_text(json.dumps({
            "settings": self.feature_settings(radius, katz_alpha),
            "n_drugs": len(drug_ids),
            "graph_nodes": len(graph), "graph_edges": graph.number_of_edges(),
        }, indent=2, default=str))
        self.last_cache_status = "miss"
        return feats

    def compute_inductive_features(
        self,
        inductive_drugs,
        ddi_edges_to_predict=None,
        radius=2,
        katz_alpha=0.005
    ):
        """
        Compute topological features for inductive drugs BEFORE removing them.
        Logic:
            1. Copy full graph (self.G)
            2. Remove the DDI edges involving inductive drugs (the edges we must predict)
            3. Compute local topo features for inductive drugs
            4. Return dataframe
        """
        print("\n[Inductive] Extracting topological features for inductive drugs...")

        # Work on a copy
        G_temp = self.G.copy()

        # --- remove DDI edges to predict ---
        if ddi_edges_to_predict is None:
            # default: remove all DDI edges with inductive drugs
            ddi_edges_to_predict = []
            for d in inductive_drugs:
                for nbr in self.G.neighbors(d):
                    # Only remove drug–drug edges
                    if self.G.nodes[nbr].get("type", "").lower() == "drug":
                        ddi_edges_to_predict.append((d, nbr))

        print(f"[Inductive] Removing {len(ddi_edges_to_predict)} DDI edges from temp graph.")
        G_temp.remove_edges_from(ddi_edges_to_predict)

        # --- compute features ---
        feats = self._compute_topo_features(
            graph=G_temp,
            drug_ids=inductive_drugs,
            radius=radius,
            katz_alpha=katz_alpha,
        )

        # optional: join GO-based features
        if self.use_protein_neighborhood and self.drug_go_df is not None:
            feats = feats.join(self.drug_go_df, how="left").fillna(0)

        print(f"[Inductive] Completed topo feature extraction for {len(feats)} drugs.\n")
        return feats

    def remove_inductive_drugs(self, inductive_drugs):
        """
        Remove inductive drugs (and incident edges) from self.G.
        Ensures they cannot influence training-time topology.
        """
        inductive_drugs = [str(d) for d in inductive_drugs]

        # Count incident edges
        incident_edges = sum(
            1 for d in inductive_drugs if d in self.G for _ in self.G.neighbors(d)
        )
        print(f"[Inductive] Removing {len(inductive_drugs)} inductive drugs...")
        print(f"[Inductive] Removing ~{incident_edges} edges incident to them...")

        # Remove nodes
        existing = [d for d in inductive_drugs if d in self.G]
        self.G.remove_nodes_from(existing)

        print(f"[Inductive] Graph now has {len(self.G)} nodes and {self.G.number_of_edges()} edges.\n")
