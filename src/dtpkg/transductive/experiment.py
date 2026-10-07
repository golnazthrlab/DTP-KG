"""Run the five-method transductive experiment from locally supplied inputs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dtpkg.data_loaders import create_folds, load_ddi_data
from dtpkg.evaluation_inputs import load_drug_depths, pair_drugs, validate_drug_depths, validate_feature_coverage
from dtpkg.evaluation_stats import comparison_table, normalize_results, performance_summary
from dtpkg.fusion.trainer import FusionTrainer
from dtpkg.models.latent_gate import Classifier, LatentGateModel
from dtpkg.project_paths import DATA_DIR
from dtpkg.topology_config import CANONICAL_GRAPH, DENSE_GRAPH_SETTINGS, make_extractor
from dtpkg.transductive.reporting import ORDER, export_figure_tables

CATEGORIES = tuple(ORDER)
MODELS = ("baseline", "topo_only", "fusion", "common_neighbors", "degree_product")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def run_transductive(
    out_dir,
    *,
    data_dir=None,
    graph_path=None,
    repeats=5,
    folds=5,
    epochs=100,
    patience=10,
    seed=42,
    device=None,
    topology_jobs=16,
    threads=None,
):
    """Fit the manuscript protocol and save private run artifacts in an empty directory.

    The defaults produce 25 split bundles (75 neural fits plus two graph
    baselines per split). The output includes record-level predictions, gates,
    splits, topology caches, and checkpoints and must remain local. Aggregate
    tables for the paper figure are exported under ``figure_data``.
    """
    for name, value, minimum in (("repeats", repeats, 1), ("folds", folds, 2),
                                  ("epochs", epochs, 1), ("patience", patience, 1),
                                  ("topology_jobs", topology_jobs, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
        raise ValueError("threads must be a positive integer or None")
    out = Path(out_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output directory must be empty: {out}")
    data = Path(DATA_DIR if data_dir is None else data_dir).expanduser().resolve()
    if graph_path is None:
        if os.environ.get("DTP_KG_NETWORK_DIR"):
            graph_path = Path(os.environ["DTP_KG_NETWORK_DIR"]).expanduser() / CANONICAL_GRAPH.name
        else:
            graph_path = CANONICAL_GRAPH if data_dir is None else data / "networks" / CANONICAL_GRAPH.name
    graph = Path(graph_path).expanduser().resolve()
    inputs = {
        "interactions/DDI_positive_pairs.csv": data / "interactions/DDI_positive_pairs.csv",
        "interactions/DDI_negative_pairs.csv": data / "interactions/DDI_negative_pairs.csv",
        "mesh/MeSH_mid_level_tfidf_svd128.csv": data / "mesh/MeSH_mid_level_tfidf_svd128.csv",
        "mesh/extended_drug_info.csv": data / "mesh/extended_drug_info.csv",
    }
    for path in [*inputs.values(), graph]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.device(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    embeddings = pd.read_csv(inputs["mesh/MeSH_mid_level_tfidf_svd128.csv"], index_col=0)
    embeddings.index = embeddings.index.astype(str)
    if embeddings.shape[1] == 0:
        raise ValueError("MeSH embeddings must have at least one column")
    validate_feature_coverage(embeddings, set(embeddings.index), "MeSH")
    X, y, pairs, positive_pool = load_ddi_data(
        level="mid_level", pos_path=inputs["interactions/DDI_positive_pairs.csv"],
        neg_path=inputs["interactions/DDI_negative_pairs.csv"], emb_dir=data / "mesh",
        seed=seed, return_positive_pool=True,
    )
    outer_folds = create_folds(X, y, n_splits=folds, seed=seed)
    drug_to_cat = load_drug_depths(inputs["mesh/extended_drug_info.csv"])
    validate_drug_depths(drug_to_cat, pair_drugs(pairs))
    package = Path(__file__).resolve().parents[1]
    source_names = (
        "transductive/experiment.py", "transductive/reporting.py", "fusion/trainer.py",
        "fusion/topology_baseline.py", "fusion/graph_baselines.py", "models/latent_gate.py",
        "topology.py", "topology_config.py", "network/graph_variants.py", "data_loaders.py",
        "ddi_labels.py", "ddi_sampling.py", "evaluation_inputs.py", "evaluation_stats.py", "metrics.py",
    )
    plan = {
        "schema_version": 1, "experiment": "transductive_fusion", "seed": seed,
        "repeats": repeats, "folds": folds, "epochs": epochs, "patience": patience,
        "device": device, "torch_threads": torch.get_num_threads(), "topology_jobs": topology_jobs,
        "models": list(MODELS), "mesh_scope": "mid_level", "mesh_dimensions": embeddings.shape[1],
        "pair_representation": "[a+b, abs(a-b)]", "positive_sampling": "per_epoch",
        "positive_to_negative_ratio": 1.0, "repeat_outer_splits": True,
        "validation_fraction": 0.15, "checkpoint_metric": "validation BCE", "min_delta": 1e-4,
        "optimizer": "Adam", "lr": 1e-3, "weight_decay": 1e-5,
        "mesh_batch_size": 128, "fusion_and_topology_batch_size": 1,
        "latent_dim": 128, "encoder_hidden": [256], "classifier_hidden": [256, 128],
        "topology_classifier_hidden": [64, 32], "dropout": 0.1,
        "gate_mode": "adaptive_vector", "gate_bias_init": 0.7,
        "drop_bio_prob": 0.0, "drop_topo_prob": 0.0,
        "neural_threshold": 0.5, "graph_threshold": "maximum inner-validation F1; ties highest threshold",
        "mesh_scaling": "StandardScaler fit on epoch-zero sampled inner-training pair features; frozen",
        "fusion_scaling": "raw per-drug MeSH and raw fold topology; learned encoders with LayerNorm",
        "topology_scaling": "mean/std of unique drugs in epoch-zero inner-training pairs; frozen, std+1e-6",
        "graph_variant": "full", "topology_settings": DENSE_GRAPH_SETTINGS,
        "heldout_edges": "remove validation and test positive DDI edges before each fold's topology and graph scores",
        "scope_of_inference": "transductive pair holdout; drugs and fixed MeSH representation may be shared across splits",
        "primary_family": "overall Fusion-MeSH AUROC (one contrast)",
        "planned_category_family": "6 categories x 2 contrasts x 5 metrics = 60 tests",
        "figure_family": "post-hoc exploratory 6 categories x 4 contrasts x 4 metrics = 96 tests",
        "python": platform.python_version(), "torch": torch.__version__,
        "input_path_base": "data_dir", "input_sha256": {name: _sha256(path) for name, path in inputs.items()},
        "graph_path": str(graph), "graph_file_sha256": _sha256(graph),
        "code_path_base": "dtpkg", "code_sha256": {name: _sha256(package / name) for name in source_names},
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_plan.json", plan)
    _write_json(out / "completion.json", {"status": "running", "split_bundles_expected": repeats * folds})
    extractor, provenance = make_extractor(
        variant="full", graph_path=graph, settings=DENSE_GRAPH_SETTINGS, backend="optimized",
        n_jobs=topology_jobs, cache_dir=out / "topo_cache", use_betweenness=False,
    )
    _write_json(out / "topo_provenance.json", provenance)
    model = LatentGateModel(
        bio_dim=embeddings.shape[1], topo_dim=extractor.num_of_topo_feats,
        latent_dim=128, fusion_mode="sym", enc_hidden=(256,), head_hidden=(256, 128),
        dropout=0.1, gate_scalar=False, gate_bias_init=0.7,
    )
    trainer = FusionTrainer(
        fusion_model_wo_go=model,
        baseline_cls=Classifier(input_dim=2 * embeddings.shape[1], hidden=(256, 128), dropout=0.1),
        topo_extractor=extractor, device=device, n_splits=folds, batch_size=128,
        epochs=epochs, lr=1e-3, weight_decay=1e-5, seed=seed, patience=patience,
        positive_sampling="per_epoch", positive_to_negative_ratio=1.0, positive_pool=positive_pool,
        repeat_outer_splits=True, include_topology_baseline=True, include_graph_baselines=True,
        out_dir=str(out),
    )
    trainer.run_experiments(X, y, pairs, outer_folds, embeddings, drug_to_cat,
                            num_experiments=repeats, checkpoint_dir=out / "checkpoints")
    results = normalize_results(trainer.results_df)
    metrics = ("auc", "f1", "prec", "rec", "acc")
    performance_summary(results, metrics, confidence=.95).to_csv(out / "performance_summary_ci.csv", index=False)
    comparison_table(results[results.category == "overall"], (("fusion", "baseline"),), ("auc",),
                     .95, family="primary_overall_AUROC").to_csv(out / "primary_overall_comparison.csv", index=False)
    comparison_table(results, (("fusion", "baseline"), ("fusion", "topo_only")), metrics, .95,
                     family="depth_categories_60", categories=CATEGORIES).to_csv(out / "category_comparisons_holm.csv", index=False)
    export_figure_tables(results, out / "figure_data", repeats=repeats, folds=folds)
    # Completion certifies that training and reporting used one unchanged set
    # of inputs and source files, even for a long-running local job.
    for name, path in inputs.items():
        if _sha256(path) != plan["input_sha256"][name]:
            raise RuntimeError(f"Input changed during training: {path}")
    if _sha256(graph) != plan["graph_file_sha256"]:
        raise RuntimeError(f"Graph changed during training: {graph}")
    for name in source_names:
        if _sha256(package / name) != plan["code_sha256"][name]:
            raise RuntimeError(f"Source changed during training: {name}")
    _write_json(out / "completion.json", {
        "status": "complete", "split_bundles": repeats * folds, "neural_fits": 3 * repeats * folds,
        "evaluated_methods": len(MODELS), "cohort_pairs": len(pairs), "cohort_drugs": len(pair_drugs(pairs)),
        "positive_training_pool": len(positive_pool),
    })
    return out
