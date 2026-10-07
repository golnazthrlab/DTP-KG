"""Table-only reporting from aggregate edge and feature-group ablation scores.

Public tables retain every prespecified hypothesis. The compact paper tables
show AUROC and F1; depth tests form their own complete Holm family. No records,
graphs, predictions or model checkpoints are needed to reconstruct inference.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dtpkg.evaluation_stats import comparison_table, performance_summary


DEPTH_CATEGORIES = (
    "low_level-low_level", "low_level-mid_level", "low_level-deep_level",
    "mid_level-mid_level", "mid_level-deep_level", "deep_level-deep_level",
)
CATEGORIES = ("overall", *DEPTH_CATEGORIES)
PRIMARY_METRICS = ("auc", "f1")
CATEGORY_METRICS = ("auc", "f1", "prec", "rec", "acc")
FEATURE_GROUPS = {
    "degree_size": ("deg_total", "deg_drug", "deg_prot", "deg_min", "deg_max",
                    "deg_mean", "deg_std", "n_nodes_subgraph"),
    "clustering_boundary": ("clust_local", "boundary_ratio"),
    "centrality": ("close_local", "katz_local"),
}
MODELS = {
    "edge_types": (
        "full_fusion", "full_topo_only", "no_ddi_fusion", "no_ddi_topo_only",
        "ddi_only_fusion", "ddi_only_topo_only", "mesh_only",
        "common_neighbors", "degree_product", "negative_count",
    ),
    "topological_features": (
        "all_descriptors", "without_degree_size", "without_clustering_boundary",
        "without_centrality",
    ),
}
CONTRASTS = {
    "edge_types": (
        ("full_fusion", "no_ddi_fusion"), ("full_fusion", "ddi_only_fusion"),
        ("full_topo_only", "no_ddi_topo_only"), ("full_topo_only", "ddi_only_topo_only"),
        ("no_ddi_fusion", "mesh_only"),
    ),
    "topological_features": tuple(
        ("all_descriptors", f"without_{group}") for group in FEATURE_GROUPS
    ),
}
FOLD_KEYS = ["model", "category", "split_repeat", "fold"]
FOLD_META = ["training_seed", "inference_protocol", "n_outer_train", "n_outer_test"]
FOLD_COLUMNS = [*FOLD_KEYS, *FOLD_META, *CATEGORY_METRICS]
TABLE_KEYS = {
    "overall": ["category", "model", "metric"],
    "comparisons": ["category", "model_a", "model_b", "metric"],
    "categories": ["category", "model", "metric"],
    "category_comparisons": ["category", "model_a", "model_b", "metric"],
}
NUMERIC_COLUMNS = {
    "estimate", "sd", "ci_low", "ci_high", "se", "n_folds", "df", "confidence",
    "p_raw", "p_holm", "family_size",
}


def _integers(values, name, minimum=0):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}") from error
    if (array.ndim != 1 or not len(array) or not np.isfinite(array).all()
            or np.any(array < minimum) or np.any(array != np.floor(array))):
        raise ValueError(f"Invalid {name}")
    return [int(x) for x in array]


def _protocol(raw, study):
    """Validate the design and retain only public numerical study metadata."""
    if study not in MODELS or raw.get("study") != study:
        raise ValueError("Unknown or mismatched ablation study")
    models = raw.get("models", [arm["name"] for arm in raw.get("arms", [])])
    if len(models) != len(set(models)) or set(models) != set(MODELS[study]):
        raise ValueError("Protocol models disagree with the complete study design")
    contrasts = [tuple(pair) for pair in raw.get("contrasts", [])]
    if contrasts != list(CONTRASTS[study]):
        raise ValueError("Protocol contrasts disagree with the prespecified family")
    if (tuple(raw.get("primary_metrics", ())) != PRIMARY_METRICS
            or tuple(raw.get("category_metrics", ())) != CATEGORY_METRICS):
        raise ValueError("Protocol must retain every prespecified metric")
    if "expected_fits" in raw:
        expected = raw["expected_fits"]
        coordinates = [(fit["arm"], fit["split_repeat"], fit["fold"]) for fit in expected]
        repetitions = sorted({repeat for _, repeat, _ in coordinates})
        folds = sorted({fold for _, _, fold in coordinates})
        if (len(coordinates) != len(set(coordinates)) or set(coordinates) != {
                (model, repeat, fold) for model in models for repeat in repetitions for fold in folds}):
            raise ValueError("Expected fits must contain the complete matched study")
    else:
        repetitions, folds = raw.get("repetitions", []), raw.get("folds", [])
    repetitions = _integers(repetitions, "repetitions")
    folds = _integers(folds, "folds")
    if len(repetitions) != len(set(repetitions)) or len(folds) != len(set(folds)):
        raise ValueError("Duplicate repetition or fold indices")
    confidence = float(raw.get("confidence", .95))
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    result = {
        "schema_version": 1, "study": study, "models": list(MODELS[study]),
        "repetitions": repetitions, "folds": folds,
        "n_fits": len(models) * len(repetitions) * len(folds),
        "contrasts": [list(pair) for pair in CONTRASTS[study]],
        "primary_metrics": list(PRIMARY_METRICS), "category_metrics": list(CATEGORY_METRICS),
        "categories": list(CATEGORIES), "confidence": confidence,
        "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "graph_variants": {
            "full": "All five canonical relation types.",
            "no_ddi": "All non-DDI relations, including protein-target and target-target edges.",
            "ddi_only": "Drug-drug interaction edges only.",
        },
        "primary_family_size": len(contrasts) * len(PRIMARY_METRICS),
        "depth_family_size": len(contrasts) * len(DEPTH_CATEGORIES) * len(CATEGORY_METRICS),
        "inference": "approximate corrected-CV; test/development size correction",
        "inferential_limit": "Conditions on the cohort and graph; shared-drug and network dependence remain.",
        "primary_threshold": "Neural models use 0.5; diagnostic thresholds use inner validation.",
    }
    for name in ("n_fits", "primary_family_size", "depth_family_size"):
        if name in raw and raw[name] != result[name]:
            raise ValueError(f"Incorrect protocol {name}")
    return result


def _fold_table(frame, protocol):
    if missing := set(FOLD_COLUMNS) - set(frame):
        raise ValueError(f"Missing aggregate fold columns: {sorted(missing)}")
    frame = frame[FOLD_COLUMNS].copy()
    if frame[FOLD_KEYS + FOLD_META].isna().any().any() or frame.duplicated(FOLD_KEYS).any():
        raise ValueError("Missing identities or duplicate aggregate fold rows")
    observed = set(frame[FOLD_KEYS].itertuples(index=False, name=None))
    expected = {(model, category, repeat, fold) for model in protocol["models"]
                for category in CATEGORIES for repeat in protocol["repetitions"]
                for fold in protocol["folds"]}
    if observed != expected:
        raise ValueError("Missing or unexpected model/category/fold rows in the declared study")
    if set(frame.inference_protocol) != {"cv"}:
        raise ValueError("Ablation reporting requires cross-validation metadata")
    for column in ("split_repeat", "fold", "training_seed", "n_outer_train", "n_outer_test"):
        frame[column] = _integers(frame[column], column, minimum=1 if column.startswith("n_outer") else 0)
    for _, group in frame.groupby(["split_repeat", "fold"]):
        if any(group[column].nunique(dropna=False) != 1 for column in FOLD_META):
            raise ValueError("Unmatched outer partition metadata or training seeds")
    for metric in CATEGORY_METRICS:
        values = pd.to_numeric(frame[metric], errors="raise").to_numpy(float)
        if np.isinf(values).any() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid aggregate {metric} scores")
        frame[metric] = values
    return frame


def calculate_tables(frame, protocol):
    """Reconstruct all performance estimates and both complete Holm families."""
    protocol = _protocol(protocol, protocol["study"])
    frame = _fold_table(frame, protocol)
    study, confidence = protocol["study"], protocol["confidence"]
    return {
        "overall": performance_summary(frame[frame.category == "overall"], PRIMARY_METRICS, confidence),
        "comparisons": comparison_table(frame, CONTRASTS[study], PRIMARY_METRICS, confidence,
                                        family=f"{study}_overall_primary", categories=["overall"]),
        "categories": performance_summary(frame[frame.category != "overall"], CATEGORY_METRICS, confidence),
        "category_comparisons": comparison_table(
            frame, CONTRASTS[study], CATEGORY_METRICS, confidence,
            family=f"{study}_depth_supplementary", categories=DEPTH_CATEGORIES),
        "fold_metrics": frame, "protocol": protocol,
    }


def _verify_table(saved, calculated, name):
    keys = TABLE_KEYS[name]
    if not set(calculated.columns).issubset(saved) or saved.duplicated(keys).any():
        raise ValueError(f"Missing columns or duplicate rows in {name}")
    saved = saved.set_index(keys).sort_index()
    calculated = calculated.set_index(keys).sort_index()
    if not saved.index.equals(calculated.index):
        raise ValueError(f"Missing or unexpected rows in {name}; retain the complete family")
    for column in calculated.columns:
        if column in NUMERIC_COLUMNS:
            if not np.allclose(saved[column].to_numpy(float), calculated[column].to_numpy(float),
                               atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError(f"Saved {name}.{column} disagrees with aggregate fold scores")
        elif not saved[column].fillna("").astype(str).equals(calculated[column].fillna("").astype(str)):
            raise ValueError(f"Saved {name}.{column} disagrees with aggregate fold scores")


def _source(source_dir):
    source = Path(source_dir).expanduser().resolve()
    return (source / "reports", source / "protocol.json") if (source / "reports").is_dir() else (source, source / "protocol.json")


def load_report(source_dir, *, study):
    """Load public tables or a completed private run and verify every saved test."""
    source, protocol_path = _source(source_dir)
    protocol = _protocol(json.loads(protocol_path.read_text()), study)
    result = calculate_tables(pd.read_csv(source / "fold_metrics.csv"), protocol)
    for name in TABLE_KEYS:
        saved = pd.read_csv(source / f"{name}.csv")
        _verify_table(saved, result[name], name)
        saved = saved[list(result[name].columns)].copy()
        # Preserve the original CSV precision after checking all numerical values.
        if "stars" in saved:
            saved["stars"] = saved.stars.fillna("")
        result[name] = saved
    return result


def export_tables(source_dir, out_dir, *, study):
    """Export a verified report using only aggregate scores and a path-free protocol.

    The destination must be outside the input directory. Record-level reports,
    private protocols and other training artifacts are never copied.
    """
    source = Path(source_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if out == source or source in out.parents:
        raise ValueError("Choose an output directory outside the input run or tables directory")
    report = load_report(source, study=study)
    out.mkdir(parents=True, exist_ok=True)
    for name in (*TABLE_KEYS, "fold_metrics"):
        report[name].to_csv(out / f"{name}.csv", index=False)
    performance_table(report).to_csv(out / "paper_performance.csv", index=False)
    contrast_table(report).to_csv(out / "paper_contrasts.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(report["protocol"], indent=2) + "\n")
    return report


def performance_table(report):
    """Return the compact paper table (fold mean ± descriptive SD)."""
    scores = report["overall"].set_index(["model", "metric"])
    study = report["protocol"]["study"]
    if study == "edge_types":
        rows = [(f"{graph}_{kind}", {"Model": label, "Graph": graph_label})
                for kind, label in (("fusion", "Latent gated fusion"), ("topo_only", "Topology only"))
                for graph, graph_label in (("full", "Full"), ("no_ddi", "No DDI"), ("ddi_only", "DDI only"))]
        rows.append(("mesh_only", {"Model": "MeSH only (original)", "Graph": "—"}))
    else:
        rows = [(name, {"Features": label}) for name, label in zip(MODELS[study], (
            "All descriptors", "Without degree/size", "Without clustering/boundary", "Without centrality"))]
    result = []
    for model, row in rows:
        for metric, label in (("auc", "AUROC"), ("f1", "F1")):
            score = scores.loc[(model, metric)]
            row[label] = f"{score.estimate:.4f} ± {score.sd:.4f}"
        result.append(row)
    return pd.DataFrame(result)


def contrast_table(report):
    """Return prespecified paired differences; stars use the complete Holm family."""
    comparisons = report["comparisons"].set_index(["model_a", "model_b", "metric"])
    study = report["protocol"]["study"]
    if study == "edge_types":
        labels = [
            {"Model": "Latent gated fusion", "Contrast": "Full − no DDI"},
            {"Model": "Latent gated fusion", "Contrast": "Full − DDI only"},
            {"Model": "Topology only", "Contrast": "Full − no DDI"},
            {"Model": "Topology only", "Contrast": "Full − DDI only"},
            {"Model": "Reference", "Contrast": "No-DDI fusion − MeSH only (original)"},
        ]
    else:
        labels = [{"Contrast": f"All − without {group}"}
                  for group in ("degree/size", "clustering/boundary", "centrality")]
    result = []
    for (model_a, model_b), row in zip(CONTRASTS[study], labels):
        for metric, label in (("auc", "ΔAUROC"), ("f1", "ΔF1")):
            score = comparisons.loc[(model_a, model_b, metric)]
            row[label] = f"{score.estimate:+.5f}{score.stars}"
        result.append(row)
    return pd.DataFrame(result)
