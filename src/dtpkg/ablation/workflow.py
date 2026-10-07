"""Matched-fold training for fusion, edge-type, and topology-feature ablations.

Preparation and preflight never fit models. Each completed arm/fold is atomic,
has an input/code identity, and can be resumed or reported independently.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch

from dtpkg.ablation.artifacts import (PACKAGE_ROOT, code_hashes, digest, file_hash, json_value, snapshot,
                               validate_completed, write_json)
from dtpkg.ablation.specification import (Arm, FEATURE_COLUMNS, FEATURE_GROUPS, StudyConfig,
                                   scientific_config, study_design)
from dtpkg.network.graph_variants import graph_content_hash, variant_provenance
from dtpkg.ddi_labels import POLICY
from dtpkg.ddi_sampling import epoch_seed
from dtpkg.evaluation_inputs import load_drug_depths, pair_drugs, validate_drug_depths, validate_feature_coverage
from dtpkg.fusion.trainer import FusionTrainer, load_cv_fit
from dtpkg.fusion.graph_baselines import graph_scores, masked_ddi_graph, validation_f1_threshold
from dtpkg.models.latent_gate import Classifier, LatentGateModel
from dtpkg.data_loaders import (_resolve_supervision_labels, build_pair_features, load_split_manifest,
                            unordered_pair_keys)
from dtpkg.topology_config import DENSE_GRAPH_SETTINGS, make_extractor
from dtpkg.fusion.topology_baseline import TopologyBaseline, TopoOnlyModel
from dtpkg.topology import TopoFeatExtractor
from dtpkg.metrics import PAIR_BINS, _bin_result_dict, _canon_pair


def _pair_records(frame):
    return [(str(a), str(b), int(y)) for a, b, y in frame[["drug1", "drug2", "label"]].itertuples(index=False, name=None)]


def _fold_dir(output, arm, rep, fold):
    return Path(output) / "fits" / arm / f"repeat_{rep:02d}_fold_{fold:02d}"


def _metrics(predictions):
    rows = []
    for category in ["overall", *PAIR_BINS]:
        sub = predictions if category == "overall" else predictions[predictions.pair_bin == category]
        y = sub.label.to_numpy(int)
        score = sub.score.to_numpy(float)
        rows.append({"category": category, **_bin_result_dict(y, score, (score >= sub.threshold.to_numpy()).astype(int))})
    return pd.DataFrame(rows)


def _predict(model, kind, pairs, embeddings, topology, device, scaler=None, normalization=None):
    """Batched, deterministic evaluation; no test-based parameter selection."""
    model = model.to(device).eval()
    validate_feature_coverage(embeddings, pair_drugs(pairs), "MeSH")
    if kind != "mesh":
        validate_feature_coverage(topology, pair_drugs(pairs), "topology")
    result = []
    with torch.no_grad():
        for start in range(0, len(pairs), 256):
            batch = pairs.iloc[start:start + 256]
            a, b = batch.drug1.tolist(), batch.drug2.tolist()
            if kind == "mesh":
                values, _, kept = build_pair_features(embeddings, batch)
                if len(kept) != len(batch):
                    raise ValueError("Prediction pairs lost feature coverage")
                logits = model(torch.as_tensor(scaler.transform(values), dtype=torch.float32, device=device))
            else:
                ta, tb = topology.loc[a].to_numpy(), topology.loc[b].to_numpy()
                if kind == "topo_only":
                    mu, sigma, columns = normalization
                    if list(topology.columns) != list(columns):
                        raise ValueError("Topology columns do not match the fitted normalization")
                    ta, tb = (ta - mu) / sigma, (tb - mu) / sigma
                    logits = model(torch.as_tensor(ta, dtype=torch.float32, device=device),
                                   torch.as_tensor(tb, dtype=torch.float32, device=device))
                else:
                    logits, _ = model(
                        torch.as_tensor(embeddings.loc[a].to_numpy(), dtype=torch.float32, device=device),
                        torch.as_tensor(ta, dtype=torch.float32, device=device),
                        torch.as_tensor(embeddings.loc[b].to_numpy(), dtype=torch.float32, device=device),
                        torch.as_tensor(tb, dtype=torch.float32, device=device))
            result.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())
    score = np.asarray(result, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("Model produced non-finite predictions")
    return score


class AblationStudy:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.output_dir
        self.arms, self.contrasts = study_design(config.study)
        self.graph = None
        self.extractors = {}
        self._reference = None
        self._reference_key = None
        self._source_predictions = None
        self._topology_tables = {}
        self._load_inputs()
        self._prepare_protocol()

    def _load_inputs(self):
        c = self.config
        self.embeddings = pd.read_csv(c.embedding_path, index_col=0)
        self.embeddings.index = self.embeddings.index.astype(str)
        self.splits, self.manifest = load_split_manifest(c.manifest_path)
        self.pairs = pd.DataFrame(self.manifest["pairs"])[["drug1", "drug2", "label"]]
        if self.manifest.get("seed") != c.seed:
            raise ValueError("The sampling seed must match the saved split manifest")
        load_split_manifest(c.manifest_path, self.pairs)
        keys = unordered_pair_keys(self.pairs)
        if len(keys) != len(set(keys)) or not set(self.pairs.label.unique()) == {0, 1}:
            raise ValueError("Evaluation cohort must contain unique unordered pairs and both classes")
        positive, negative = _resolve_supervision_labels(
            pd.read_csv(c.positive_path), pd.read_csv(c.negative_path), POLICY, None)
        ids = set(self.embeddings.index)
        self.positive_pool = positive[positive.drug1.isin(ids) & positive.drug2.isin(ids)].copy().assign(label=1)
        negative = negative[negative.drug1.isin(ids) & negative.drug2.isin(ids)]
        source_keys = {1: set(unordered_pair_keys(self.positive_pool)), 0: set(unordered_pair_keys(negative))}
        for label in (0, 1):
            if not set(unordered_pair_keys(self.pairs[self.pairs.label == label])).issubset(source_keys[label]):
                raise ValueError("Manifest labels do not match the current resolved supervision")
        self.depths = load_drug_depths(c.depth_path)
        self.all_drugs = sorted(pair_drugs(self.pairs) | pair_drugs(self.positive_pool))
        validate_feature_coverage(self.embeddings, self.all_drugs, "MeSH")
        validate_drug_depths(self.depths, pair_drugs(self.pairs))
        reps = list(range(len(self.splits))) if c.repetitions is None else sorted(c.repetitions)
        self.coordinates = []
        outer_partitions = set()
        for rep in reps:
            if rep >= len(self.splits):
                raise ValueError(f"Unknown split repetition {rep}")
            outer = tuple(tuple(sorted(map(int, test))) for _, _, test in self.splits[rep])
            if outer in outer_partitions:
                raise ValueError("Repeated identical outer splits are training repetitions, not repeated CV")
            outer_partitions.add(outer)
            folds = list(range(len(self.splits[rep]))) if c.folds is None else sorted(c.folds)
            for fold in folds:
                if fold >= len(self.splits[rep]):
                    raise ValueError(f"Unknown fold {fold}")
                tr, va, te = self.splits[rep][fold]
                if set(tr) | set(va) | set(te) != set(range(len(self.pairs))):
                    raise ValueError("Split does not partition the complete evaluation cohort")
                for idx in (tr, va, te):
                    if len(idx) != len(set(idx)) or set(self.pairs.iloc[idx].label) != {0, 1}:
                        raise ValueError("Each partition needs unique rows and both classes")
                self.coordinates.append((rep, fold))
        self.source_paths = {name: Path(value) for name, value in {
            "graph": c.graph_path, "positives": c.positive_path, "negatives": c.negative_path,
            "embeddings": c.embedding_path, "depths": c.depth_path}.items()}
        self.source_hashes = {name: file_hash(path) for name, path in self.source_paths.items()}
        self.source_stats = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in self.source_paths.items()}
        self.manifest_hash = file_hash(c.manifest_path)

    def _prepare_protocol(self):
        c = self.config
        self.fit_code = code_hashes(PACKAGE_ROOT)
        self.data_identity = digest({"files": self.source_hashes, "pairs": _pair_records(self.pairs)})
        reference_paths = []
        if c.reuse_reference:
            reference_paths = [Path(c.reference_dir) / name for name in
                               ("split_manifest.json", "topo_provenance.json", "oof_predictions.csv")]
            reference_paths += [Path(c.reference_dir) / "checkpoints" / f"fit_{rep:02d}_{fold:02d}.pt"
                                for rep, fold in self.coordinates]
        self.reference_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                                for path in reference_paths}
        self.protocol = dict(
            format_version=1, study=c.study, scientific_config=scientific_config(c),
            environment={"python": sys.version.split()[0], "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "networkx": nx.__version__},
            data_identity=self.data_identity, split_manifest_hash=self.manifest_hash,
            reference_sources={str(path): file_hash(path) for path in reference_paths},
            source_files={name: {"path": str(path), "sha256": self.source_hashes[name]}
                          for name, path in self.source_paths.items()},
            code_path_base="dtpkg", code_hashes=self.fit_code, coordinates=self.coordinates,
            arms=[asdict(arm) for arm in self.arms], contrasts=self.contrasts,
            primary_metrics=["auc", "f1", "rec"] if c.study == "sampling_policy" else ["auc", "f1"],
            category_metrics=["auc", "f1", "prec", "rec", "acc"], confidence=c.confidence,
            expected_fits=[dict(arm=arm.name, split_repeat=rep, fold=fold)
                           for rep, fold in self.coordinates for arm in self.arms],
            pairing="sum_and_absolute_difference", positive_to_negative_ratio=1.0,
            graph_policy="all canonical known DDIs except validation and test positive targets; fixed across epochs",
            topology_settings={**DENSE_GRAPH_SETTINGS, "radius": 2, "use_betweenness": False},
            feature_groups=FEATURE_GROUPS,
            topology_only_scaling="mean/std(+1e-6) of unique epoch-zero training endpoints",
            fusion_scaling_sensitivity="mean/std(+1e-6) of unique epoch-zero training endpoints; frozen",
            threshold_policy="neural .5 primary; validation-F1-selected secondary; graph diagnostics validation-selected",
            inferential_limit="Approximate corrected-CV inference does not fully account for shared drugs/network dependence",
            checkpoint_policy="select by inner-validation BCE; outer test never selects settings",
            fusion_initialization="shared encoders/head from seeded adaptive-vector initialization; preserve its post-init RNG state",
        )
        self.protocol["protocol_id"] = digest(self.protocol)
        self.protocol = json_value(self.protocol)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != self.protocol:
            raise ValueError(f"Configuration, inputs or code changed for {self.output_dir}; use a new results_root")
        write_json(path, self.protocol)
        copied_manifest = self.output_dir / "split_manifest.json"
        if copied_manifest.exists() and file_hash(copied_manifest) != self.manifest_hash:
            raise ValueError("Saved ablation split manifest changed")
        if not copied_manifest.exists():
            shutil.copyfile(c.manifest_path, copied_manifest)
        snapshot(PACKAGE_ROOT, self.output_dir / "code_snapshot.zip", self.fit_code)

    def _identity(self, arm, rep, fold):
        return {key: self.protocol[key] for key in ("protocol_id", "data_identity", "split_manifest_hash")} | {
            "arm": arm.name, "split_repeat": rep, "fold": fold}

    def _assert_unchanged(self):
        if code_hashes(PACKAGE_ROOT) != self.fit_code:
            raise RuntimeError("Fitting code changed after preparation. Restart the kernel and use a new results_root.")
        if file_hash(self.config.manifest_path) != self.manifest_hash:
            raise RuntimeError("Split manifest changed after preparation")
        for name, path in self.source_paths.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != self.source_stats[name]:
                raise RuntimeError(f"Input changed after preparation: {path}")
        for name, expected in self.reference_stats.items():
            path = Path(name)
            if (path.stat().st_size, path.stat().st_mtime_ns) != expected:
                raise RuntimeError(f"Reference artifact changed after preparation: {path}")

    def plan_table(self):
        rows = []
        for arm in self.arms:
            completed = sum(validate_completed(_fold_dir(self.output_dir, arm.name, rep, fold),
                                              self._identity(arm, rep, fold)) for rep, fold in self.coordinates)
            rows.append({**asdict(arm), "fits": len(self.coordinates), "completed": completed,
                         "action": "diagnostic" if arm.reference_only and arm.model_kind != "mesh" else
                         "reuse verified reference" if arm.reference_model and self.config.reuse_reference else "fit"})
        return pd.DataFrame(rows)

    def _load_graph(self):
        if self.graph is None:
            self.graph = nx.read_graphml(self.config.graph_path)
            if self.graph.is_directed() or self.graph.is_multigraph() or nx.number_of_selfloops(self.graph):
                raise ValueError("Ablations require a simple undirected loop-free canonical graph")
            missing = set(self.all_drugs) - set(self.graph)
            if missing:
                raise ValueError(f"Canonical graph lacks requested drugs: {sorted(missing)[:10]}")
            if any(str(self.graph.nodes[d].get("type", "")).lower() != "drug" for d in self.all_drugs):
                raise ValueError("Requested drug nodes have inconsistent graph types")
            self.base_provenance = variant_provenance(self.graph, "full")
            self.target_drugs = {d for d in self.all_drugs if any(
                str(self.graph.nodes[n].get("type", "")).lower() != "drug" for n in self.graph[d])}
            negatives = self.pairs[self.pairs.label == 0]
            if any(self.graph.has_edge(a, b) for a, b in zip(negatives.drug1, negatives.drug2)):
                raise ValueError("An evaluation negative is a canonical graph DDI")
            if any(not self.graph.has_edge(a, b) for a, b in zip(self.positive_pool.drug1, self.positive_pool.drug2)):
                raise ValueError("Training positive reservoir is inconsistent with canonical DDI edges")
        return self.graph

    def _partitions(self, rep, fold):
        return tuple(self.pairs.iloc[idx].reset_index(drop=True) for idx in self.splits[rep][fold])

    def _trainer(self, arm, out_dir):
        c = self.config
        return FusionTrainer(
            device=c.device, epochs=c.epochs, patience=c.patience, min_delta=c.min_delta,
            lr=c.lr, weight_decay=c.weight_decay, batch_size=c.batch_size, seed=c.seed,
            val_frac=self.manifest["val_frac"], positive_sampling=arm.positive_sampling,
            positive_to_negative_ratio=1.0, positive_pool=self.positive_pool,
            out_dir=str(out_dir), drop_bio_prob=0.0, drop_topo_prob=0.0,
            baseline_cls=Classifier(2 * self.embeddings.shape[1], hidden=c.head_hidden, dropout=c.dropout))

    def _reference_bundle(self, rep, fold):
        if self._reference_key == (rep, fold):
            return self._reference
        c = self.config
        path = Path(c.reference_dir) / "checkpoints" / f"fit_{rep:02d}_{fold:02d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Reference checkpoint missing: {path}; disable reuse_reference to refit")
        bundle = load_cv_fit(path)
        train, val, test = self._partitions(rep, fold)
        for key, frame in (("train_pairs", train), ("validation_pairs", val), ("test_pairs", test)):
            if _pair_records(bundle[key]) != _pair_records(frame):
                raise ValueError(f"Reference {path.name} has different {key}")
        expected_config = dict(epochs=c.epochs, patience=c.patience, min_delta=c.min_delta,
                               lr=c.lr, weight_decay=c.weight_decay, batch_size=c.batch_size,
                               drop_bio_prob=0.0, drop_topo_prob=0.0, val_frac=self.manifest["val_frac"])
        if bundle["trainer_config"] != expected_config:
            raise ValueError(f"Reference trainer settings differ for {path.name}; disable reuse_reference or restore settings")
        sampling = bundle["sampling_config"]
        expected_sampling = dict(seed=c.seed, fold=fold, repetition=rep, mode="per_epoch", positive_to_negative_ratio=1.0)
        if any(sampling.get(k) != v for k, v in expected_sampling.items()):
            raise ValueError("Reference sampling settings differ")
        heldout_positive_count = int(val.label.sum() + test.label.sum())
        if sampling.get("training_positive_pool") != len(self.positive_pool) - heldout_positive_count:
            raise ValueError("Reference positive reservoir size differs")
        if bundle["metadata"].get("training_seed") != epoch_seed(c.seed, fold, rep, 0):
            raise ValueError("Reference fit seed differs")
        if not bundle["embeddings"].equals(self.embeddings):
            raise ValueError("Reference MeSH representation differs")
        validate_feature_coverage(bundle["eval_topology"], self.all_drugs, "reference topology")
        if set(bundle["eval_topology"].columns) != set(FEATURE_COLUMNS):
            raise ValueError("Reference uses a different topology feature inventory")
        expected = dict(bio_dim=self.embeddings.shape[1], topo_dim=12, latent_dim=c.latent_dim,
                        fusion_mode="sym", enc_hidden=c.enc_hidden, head_hidden=c.head_hidden,
                        dropout=c.dropout, gate_scalar=False, gate_bias_init=0.7)
        init = dict(bundle["models"]["fusion"]._init_kwargs)
        if init.pop("gate_mode", "adaptive_vector") != "adaptive_vector" or json_value(init) != json_value(expected):
            raise ValueError("Reference fusion architecture differs")
        topo = bundle["models"]["topo_only"]
        widths = [m.out_features for m in topo.net if isinstance(m, torch.nn.Linear)]
        if widths != [*c.topo_hidden, 1] or getattr(topo, "pair_mode", None) != "sym":
            raise ValueError("Reference topology-only architecture differs")
        mesh = bundle["models"]["baseline"]
        if [m.out_features for m in mesh.net if isinstance(m, torch.nn.Linear)] != [*c.head_hidden, 1]:
            raise ValueError("Reference MeSH architecture differs")
        for model in (mesh, topo):
            if any(m.p != c.dropout for m in model.modules() if isinstance(m, torch.nn.Dropout)):
                raise ValueError("Reference dropout differs")
        self._reference_key, self._reference = (rep, fold), bundle
        return bundle

    def preflight(self):
        """Check inputs, graph and reusable fits; never extract topology or train."""
        self._assert_unchanged()
        self._load_graph()
        checks = [dict(check="resolved cohort", status="ok", detail=f"{len(self.pairs)} fixed pairs"),
                  dict(check="matched partitions", status="ok", detail=f"{len(self.coordinates)} selected CV folds"),
                  dict(check="feature coverage", status="ok", detail=f"{len(self.all_drugs)} eligible endpoints"),
                  dict(check="canonical graph", status="ok", detail=self.base_provenance["graph_content_hash"])]
        if self.config.reuse_reference:
            provenance_path = Path(self.config.reference_dir) / "topo_provenance.json"
            provenance = json.loads(provenance_path.read_text())
            if provenance["graph_content_hash"] != self.base_provenance["graph_content_hash"] or provenance["variant"] != "full":
                raise ValueError("Reference graph differs from the selected canonical graph")
            settings = provenance["feature_settings"]
            expected = TopoFeatExtractor(graph=nx.Graph(), backend="optimized",
                                         use_betweenness=False, **DENSE_GRAPH_SETTINGS).feature_settings(
                                             radius=2, katz_alpha=0.005)
            if any(settings.get(k) != v for k, v in expected.items()):
                raise ValueError("Reference topology settings differ")
            for rep, fold in self.coordinates:
                self._reference_bundle(rep, fold)
            checks.append(dict(check="reference compatibility", status="ok", detail="all selected checkpoints verified"))
        else:
            checks.append(dict(check="reference reuse", status="disabled", detail="reference arms will be fitted"))
        report = pd.DataFrame(checks)
        report.to_csv(self.output_dir / "preflight.csv", index=False)
        write_json(self.output_dir / "graph_provenance.json", self.base_provenance)
        self._preflight_ok = True
        return report

    def _extractor(self, variant):
        if variant not in self.extractors:
            extractor, provenance = make_extractor(
                variant=variant, graph=self._load_graph(), settings=DENSE_GRAPH_SETTINGS,
                backend="optimized", n_jobs=self.config.n_jobs,
                cache_dir=Path(self.config.results_root) / "_topology_cache" / variant,
                use_betweenness=False)
            self.extractors[variant] = extractor
            write_json(self.output_dir / "graphs" / variant / "provenance.json", provenance)
        return self.extractors[variant]

    def _topology(self, variant, rep, fold):
        key = (variant, rep, fold)
        if key in self._topology_tables:
            return self._topology_tables[key]
        _, val, test = self._partitions(rep, fold)
        heldout = pd.concat([val, test], ignore_index=True)
        if variant == "full" and self.config.reuse_reference:
            frame = self._reference_bundle(rep, fold)["eval_topology"].loc[self.all_drugs].copy()
            origin = "verified_reference_checkpoint"
        else:
            extractor = self._extractor(variant)
            frame = extractor.compute_for_fold(self.all_drugs, eval_pairs=heldout)
            diag = extractor.last_diagnostics
            if (not {"drugbank_id", "status", "katz_status"}.issubset(diag.columns)
                    or diag.drugbank_id.duplicated().any()
                    or set(diag.drugbank_id) != set(self.all_drugs)):
                raise ValueError("Topology diagnostics are missing/incomplete; remove the affected topology cache and retry")
            if not diag.status.eq("computed").all():
                raise ValueError("Topology extraction failed; no missing-node zero imputation is permitted")
            if not diag.katz_status.eq("converged").all():
                raise ValueError("Katz failed under the declared dense-graph settings")
            diag.to_csv(self.output_dir / "graphs" / variant / f"diagnostics_{rep:02d}_{fold:02d}.csv", index=False)
            origin = "masked_graph_extraction"
        validate_feature_coverage(frame, self.all_drugs, "topology")
        if set(frame.columns) != set(FEATURE_COLUMNS):
            raise ValueError("Unexpected topology schema; feature ablations require the declared 12 descriptors")
        frame = frame.loc[self.all_drugs]
        # Dataframe operations in each arm must not alter the shared raw table.
        self._topology_tables[key] = frame
        removed = heldout[heldout.label == 1]
        removed_edges = len(removed) if variant != "no_ddi" else 0
        if variant == "full":
            count = self.graph.number_of_edges()
            edge_types = dict(self.base_provenance["edges_by_endpoint_type"])
        else:
            selected = self._extractor(variant).G
            count = selected.number_of_edges()
            edge_types = dict(self.base_provenance["edges_by_endpoint_type"])
            if variant == "no_ddi":
                edge_types.pop("drug-drug", None)
            else:
                edge_types = {"drug-drug": edge_types["drug-drug"]}
        if removed_edges:
            edge_types["drug-drug"] -= removed_edges
        write_json(self.output_dir / "graphs" / variant / f"mask_{rep:02d}_{fold:02d}.json", {
            "origin": origin, "graph_variant": variant, "split_repeat": rep, "fold": fold,
            "masked_positive_pairs": _pair_records(removed), "masked_edge_count": removed_edges,
            "edges_after_mask": count - removed_edges, "node_count": len(self.graph),
            "edges_by_endpoint_type_after_mask": edge_types,
            "isolated_requested_drugs": int(frame.deg_total.eq(0).sum()),
            "feature_columns": list(frame.columns), "requested_drugs": len(frame),
            "constant_columns": [col for col in frame if frame[col].nunique() <= 1]})
        return frame

    def _fusion_factory(self, topo_dim, mode):
        c = self.config
        kwargs = dict(bio_dim=self.embeddings.shape[1], topo_dim=topo_dim, latent_dim=c.latent_dim,
                      enc_hidden=c.enc_hidden, head_hidden=c.head_hidden, dropout=c.dropout,
                      fusion_mode="sym", gate_bias_init=0.7)
        def factory():
            rng = torch.get_rng_state()
            reference = LatentGateModel(**kwargs)
            after = torch.get_rng_state()
            if mode == "adaptive_vector":
                return reference
            torch.set_rng_state(rng)
            model = LatentGateModel(**kwargs, gate_mode=mode)
            for name in ("bio_encoder", "topo_encoder", "classifier"):
                getattr(model, name).load_state_dict(getattr(reference, name).state_dict())
            torch.set_rng_state(after)
            return model
        return factory

    def _metadata(self, arm, rep, fold):
        tr, va, te = self.splits[rep][fold]
        return dict(model=arm.name, split_repeat=rep, fold=fold,
                    training_seed=epoch_seed(self.config.seed, fold, rep, 0), inference_protocol="cv",
                    n_outer_train=len(tr) + len(va), n_outer_test=len(te))

    def _annotate(self, pairs, scores, arm, rep, fold, threshold, topology):
        frame = pairs.copy()
        frame["score"] = scores
        frame["threshold"] = threshold
        frame["threshold_policy"] = "fixed_0.5" if arm.model_kind in ("fusion", "mesh", "topo_only") else "inner_validation_F1"
        frame["pair_bin"] = [_canon_pair(self.depths[a], self.depths[b]) for a, b in zip(frame.drug1, frame.drug2)]
        train, _, _ = self._partitions(rep, fold)
        roster = pair_drugs(train[train.label == 0])
        for i in (1, 2):
            ids = frame[f"drug{i}"]
            frame[f"has_target{i}"] = ids.isin(self.target_drugs)
            frame[f"in_training_negative_roster{i}"] = ids.isin(roster)
            frame[f"isolated{i}"] = ids.map(topology.deg_total.eq(0))
        return frame.assign(**self._metadata(arm, rep, fold))

    def _diagnostic(self, arm, rep, fold):
        train, val, test = self._partitions(rep, fold)
        if arm.model_kind == "negative_count":
            negatives = train[train.label == 0]
            counts = Counter(negatives.drug1.tolist() + negatives.drug2.tolist())
            score = lambda frame: np.array([-counts[a] - counts[b] for a, b in zip(frame.drug1, frame.drug2)], dtype=float)
        else:
            graph = masked_ddi_graph(self.graph, pd.concat([val, test], ignore_index=True))
            score = lambda frame: graph_scores(graph, frame, arm.model_kind)
        vp, tp = score(val), score(test)
        threshold = validation_f1_threshold(val.label, vp)
        return vp, tp, threshold

    def _source_test_scores(self, arm, rep, fold, computed):
        path = Path(self.config.reference_dir) / "oof_predictions.csv"
        if not path.exists():
            return computed
        if self._source_predictions is None:
            self._source_predictions = pd.read_csv(path)
        source = self._source_predictions
        selected = source[(source.model == arm.reference_model) & (source.split_repeat == rep) & (source.fold == fold)]
        _, _, test = self._partitions(rep, fold)
        if _pair_records(selected) != _pair_records(test):
            raise ValueError("Reference predictions have different pair identities/order")
        column = "score" if "score" in selected else "pred"
        saved = selected[column].to_numpy(float)
        if not np.allclose(computed, saved, rtol=2e-5, atol=2e-6):
            raise ValueError("Reference checkpoint replay does not reproduce saved scores")
        return saved  # preserve the exact published-reference scores after replay verification

    def _fit_one(self, arm, rep, fold, folder):
        c = self.config
        train, val, test = self._partitions(rep, fold)
        raw = self._topology(arm.graph_variant, rep, fold)
        topology = raw.copy()
        if arm.feature_group != "all":
            topology = topology.drop(columns=list(FEATURE_GROUPS[arm.feature_group]))
        trainer = self._trainer(arm, folder)
        trainer.drug_to_cat = self.depths
        sampler = trainer.make_sampler(train, pd.concat([val, test], ignore_index=True), self.embeddings, fold, rep)
        fit_seed = epoch_seed(c.seed, fold, rep, 0)
        model, scaler, normalization, fusion_scaler = None, None, None, None
        gates = pd.DataFrame(columns=["drug_id", "gate_mean", "gate_sd_dimensions"])
        started = time.monotonic()
        threshold = 0.5
        reused = arm.reference_model is not None and c.reuse_reference
        if arm.model_kind in ("negative_count", "common_neighbors", "degree_product"):
            vp, tp, threshold = self._diagnostic(arm, rep, fold)
            info = dict(epochs_run=0, best_epoch=0, source="diagnostic", n_train=len(train), n_val=len(val), n_test=len(test))
            history = []
        else:
            if reused:
                bundle = self._reference_bundle(rep, fold)
                model = deepcopy(bundle["models"][arm.reference_model])
                scaler = bundle["baseline_scaler"] if arm.model_kind == "mesh" else None
                normalization = bundle["topo_only_normalization"] if arm.model_kind == "topo_only" else None
                info = deepcopy(bundle["fit_info"][arm.reference_model])
                history = deepcopy(bundle["epoch_class_losses"][arm.reference_model])
                info["source"] = "verified_reference_checkpoint"
                info["reference_path"] = str(Path(c.reference_dir) / "checkpoints" / f"fit_{rep:02d}_{fold:02d}.pt")
                info["reference_sha256"] = file_hash(info["reference_path"])
            else:
                if arm.topology_scaling == "train_standardized":
                    scaling_drugs = sorted(pair_drugs(sampler.epoch(0)))
                    mu = topology.loc[scaling_drugs].mean(axis=0)
                    sigma = topology.loc[scaling_drugs].std(axis=0, ddof=0) + 1e-6
                    fusion_scaler = dict(mean=mu, scale=sigma, training_drugs=scaling_drugs)
                    topology = (topology - mu) / sigma
                if arm.model_kind == "fusion":
                    _, model = trainer._train_fusion(
                        self._fusion_factory(topology.shape[1], arm.gate_mode), train, val, topology,
                        self.embeddings, self.depths, epoch_sampler=sampler, fit_seed=fit_seed)
                elif arm.model_kind == "topo_only":
                    ablation = TopologyBaseline(trainer)
                    ablation._train_topo_only(TopoOnlyModel(topology.shape[1], hidden=c.topo_hidden, dropout=c.dropout),
                                             train, val, topology, epoch_sampler=sampler, fit_seed=fit_seed,
                                             device=c.device, lr=c.lr, weight_decay=c.weight_decay)
                    model, normalization = ablation.last_model, ablation.last_normalization
                elif arm.model_kind == "mesh":
                    train_loader, val_loader, _, epoch_loader, scaler = trainer.sampled_loaders(sampler, self.embeddings, val)
                    _, model = trainer._train_baseline(train_loader, val_loader, epoch_sampler=sampler,
                                                       epoch_loader=epoch_loader, fit_seed=fit_seed)
                else:
                    raise ValueError(f"Unknown model kind {arm.model_kind}")
                info, history = deepcopy(trainer.last_fit_info), deepcopy(trainer.last_epoch_losses)
                info["source"] = "fitted"
            vp = _predict(model, arm.model_kind, val, self.embeddings, topology, c.device, scaler, normalization)
            tp = _predict(model, arm.model_kind, test, self.embeddings, topology, c.device, scaler, normalization)
            swapped = test.rename(columns={"drug1": "drug2", "drug2": "drug1"})
            reversed_scores = _predict(model, arm.model_kind, swapped, self.embeddings, topology, c.device, scaler, normalization)
            difference = float(np.max(np.abs(tp - reversed_scores)))
            if difference > 2e-6:
                raise ValueError(f"Pair-swap invariance failed: {difference}")
            info["max_swap_score_difference"] = difference
            if reused:
                tp = self._source_test_scores(arm, rep, fold, tp)
            info["unique_positive_exposure_selected_checkpoint"] = sampler.unique_positives(int(info["best_epoch"]))
            info["unique_positive_exposure_stopped_epoch"] = sampler.unique_positives(int(info["epochs_run"]))
            info["parameter_count"] = sum(p.numel() for p in model.parameters())
            if arm.model_kind == "fusion":
                drugs = sorted(pair_drugs(test))
                with torch.no_grad():
                    _, values = model.fuse_one(
                        torch.as_tensor(self.embeddings.loc[drugs].to_numpy(), dtype=torch.float32, device=c.device),
                        torch.as_tensor(topology.loc[drugs].to_numpy(), dtype=torch.float32, device=c.device))
                if values is not None:
                    values = values.detach().cpu().numpy()
                    gates = pd.DataFrame(dict(drug_id=drugs, gate_mean=values.mean(1),
                                              gate_sd_dimensions=values.std(1)))
            checkpoint = dict(format_version=1, protocol="ablation", identity=self._identity(arm, rep, fold),
                              arm=asdict(arm), model=deepcopy(model).cpu(), scaler=scaler,
                              normalization=normalization, fusion_scaler=fusion_scaler,
                              topology=topology, embeddings=self.embeddings, depths=self.depths,
                              train_pairs=train, validation_pairs=val, test_pairs=test,
                              fit_info=info, sampling=sampler.describe(), config=scientific_config(c))
            temporary = folder / "checkpoint.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(folder / "checkpoint.pt")
        info.update(wall_seconds=time.monotonic() - started, **asdict(arm), **self._metadata(arm, rep, fold))
        validation = self._annotate(val, vp, arm, rep, fold, threshold, raw)
        predictions = self._annotate(test, tp, arm, rep, fold, threshold, raw)
        predictions.to_csv(folder / "predictions.csv", index=False)
        validation.to_csv(folder / "validation_predictions.csv", index=False)
        _metrics(predictions).assign(**self._metadata(arm, rep, fold)).to_csv(folder / "fold_metrics.csv", index=False)
        pd.DataFrame(history, columns=None if history else ["epoch"]).to_csv(folder / "history.csv", index=False)
        gates.assign(phase="trained", **self._metadata(arm, rep, fold)).to_csv(folder / "gates.csv", index=False)
        write_json(folder / "fit_info.json", info)
        write_json(folder / "sampling.json", sampler.describe())

    def run(self):
        """Fit/reuse selected arms; resume only exact, intact completed fits."""
        if not getattr(self, "_preflight_ok", False):
            self.preflight()
        torch.set_num_threads(1)
        self._assert_unchanged()
        for rep, fold in self.coordinates:
            for arm in self.arms:
                folder = _fold_dir(self.output_dir, arm.name, rep, fold)
                identity = self._identity(arm, rep, fold)
                if validate_completed(folder, identity):
                    print(f"Reuse completed {arm.name}: repeat {rep}, fold {fold}")
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                lock = folder / ".running"
                try:
                    descriptor = lock.open("x")
                except FileExistsError:
                    raise RuntimeError(f"Fit is already running or was interrupted: {lock}. Remove the lock only after confirming its process stopped.") from None
                try:
                    descriptor.write(f"pid={__import__('os').getpid()}\n"); descriptor.close()
                    self._assert_unchanged()
                    print(f"\n{arm.name}: repeat {rep}, fold {fold}", flush=True)
                    self._fit_one(arm, rep, fold, folder)
                    self._assert_unchanged()
                    files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".tmp")]
                    write_json(folder / "completed.json", {**identity,
                               "artifact_hashes": {p.name: file_hash(p) for p in files}})
                finally:
                    lock.unlink(missing_ok=True)
            # Keep memory bounded to one fold, not all 25 masked feature tables.
            self._topology_tables.clear()
            self._reference = self._reference_key = None
        write_json(self.output_dir / "completion.json", {
            **{k: self.protocol[k] for k in ("protocol_id", "data_identity", "split_manifest_hash")},
            "completed_fits": len(self.protocol["expected_fits"])})
        return self.plan_table()


def prepare_study(config=None, **kwargs):
    """Load fixed inputs and write an inspectable protocol; no model fitting."""
    if config is not None and kwargs:
        raise TypeError("Supply StudyConfig or keyword settings, not both")
    return AblationStudy(config if config is not None else StudyConfig(**kwargs))


def score_checkpoint(path, pairs=None, device="cpu"):
    """Replay a trusted local ablation checkpoint without fitting or new transforms."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != "ablation" or bundle.get("format_version") != 1:
        raise ValueError("Expected a version-1 ablation checkpoint")
    frame = bundle["test_pairs"] if pairs is None else pairs
    score = _predict(bundle["model"], bundle["arm"]["model_kind"], frame,
                     bundle["embeddings"], bundle["topology"], device,
                     bundle["scaler"], bundle["normalization"])
    return frame.reset_index(drop=True).assign(score=score)


def report_study(output_dir, allow_partial=False):
    """Notebook convenience entry point; reporting reads saved artifacts only."""
    from dtpkg.ablation.reporting import report_study as report
    return report(output_dir, allow_partial=allow_partial)
