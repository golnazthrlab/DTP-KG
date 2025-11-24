import os
import pandas as pd
import networkx as nx
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import entropy
import time


class TopoFeatExtractor:
    def __init__(self,
                 graph_path: str,
                 use_betweenness: bool = False,
                 use_protein_neighborhood: bool = False,
                 protein_feat_path: str = "datasets/drug_GO_neighborhood_features.csv",
                 use_weighted: bool = False,
                 ):

        self.G = nx.read_graphml(graph_path)

        node_types = nx.get_node_attributes(self.G, "type")
        self.drug_nodes = [n for n, t in node_types.items() if t.lower() == "drug"]

        print(f"   Graph loaded: {len(self.G)} nodes, {self.G.number_of_edges()} edges")
        print(f"   Found {len(self.drug_nodes)} drug nodes.\n")

        # Core settings
        self.use_betweenness = use_betweenness
        self.use_protein_neighborhood = use_protein_neighborhood

        # --- Load precomputed drug-level GO features if requested ---
        if use_protein_neighborhood:
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



    def compute_local_features(self, G, node, radius=2, katz_alpha=0.005):
        weight_arg = "weight" if self.use_weighted else None

        # --- ego graph ---
        t0 = time.time()
        if self.use_weighted:
            subG = nx.ego_graph(G, node, radius=radius, distance="weight")
        else:
            subG = nx.ego_graph(G, node, radius=radius)
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
        if len(subG) < 200:
            close_local = nx.closeness_centrality(subG, distance=weight_arg).get(node, 0)
        else:
            close_local = 0
        # print(f"[{node}] closeness: {time.time() - t5:.4f}s")

        # --- Katz ---
        t6 = time.time()
        try:
            katz_local = nx.katz_centrality(
                subG, alpha=katz_alpha, max_iter=100, tol=1e-3, weight=weight_arg
            ).get(node, 0)

        except Exception:
            katz_local = 0
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


    def _compute_topo_features(self, graph, drug_ids, radius=2, katz_alpha=0.005):
        """Compute local features for a list of drug IDs."""
        rows = []
        t0 = time.time()
        for i, d in enumerate(drug_ids):
            if d not in graph:
                continue
            try:
                feats = self.compute_local_features(graph, d, radius, katz_alpha)
                row = {"drugbank_id": d}
                row.update(feats)
                rows.append(row)
            except Exception as e:
                print(f"Error computing {d}: {e}")

            if (i + 1) % 500 == 0:
                print(f"→ Processed {i+1}/{len(drug_ids)} drugs ({time.time()-t0:.1f}s elapsed)")

        if not rows:
            print("No valid topological features computed — returning empty DataFrame.")
            return pd.DataFrame(columns=[
                "deg_total", "deg_drug", "deg_prot",
                "deg_min", "deg_max", "deg_mean", "deg_std",
                "clust_local", "boundary_ratio",
                "close_local", "katz_local", "betw_local", "n_nodes_subgraph"
            ])

        df = pd.DataFrame(rows).set_index("drugbank_id")
        print(f"Finished {len(df)} drugs in {time.time()-t0:.1f}s total.")
        return df
       
 

    def compute_for_fold(self, 
                         drug_ids, 
                         eval_pairs=None, 
                         radius=2, 
                         katz_alpha=0.005,
                    ):
        """Compute fold-specific topological features (with optional edge removal)."""
        edges_to_remove = []
        if eval_pairs is not None:
            for _, r in eval_pairs.iterrows():
                if r.get("label", 1) == 1:
                    edges_to_remove.append((str(r["drug1"]), str(r["drug2"])))

        print(f"Removing {len(edges_to_remove)} eval DDI edges for leakage safety.")
        self.G.remove_edges_from(edges_to_remove)
        print(f"Computing topological features for {len(drug_ids)} drugs...")
        try:
            feats = self._compute_topo_features(self.G, drug_ids, radius, katz_alpha)
            if self.use_protein_neighborhood and self.drug_go_df is not None:
                feats = feats.join(self.drug_go_df, how="left").fillna(0)
        finally:
            self.G.add_edges_from(edges_to_remove)

            
        print(f"Computed features for {len(feats)} drugs.")
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
        Permanently remove inductive drugs from the working graph self.G.
        This prevents them from influencing training/validation topo features.
        """
        print(f"[Inductive] Removing {len(inductive_drugs)} inductive drugs from training graph...")

        # Ensure all exist as strings (GraphML loads node ids as strings)
        inductive_drugs = [str(d) for d in inductive_drugs]

        # --- Count incident edges BEFORE removal ---
        incident_edges = set()
        for d in inductive_drugs:
            if d in self.G:
                for nbr in self.G.neighbors(d):
                    # store edge as a sorted tuple to avoid duplicates
                    edge = tuple(sorted((d, nbr)))
                    incident_edges.add(edge)

        print(f"[Inductive] Found {len(incident_edges)} edges incident to inductive drugs.")
        print(f"[Inductive] Removing {len(inductive_drugs)} inductive drugs from training graph...")

        # Remove nodes + all edges incident to them
        for d in inductive_drugs:
            if d in self.G:
                self.G.remove_node(d)

        print(f"[Inductive] Graph now has {len(self.G)} nodes, {self.G.number_of_edges()} edges.\n")    