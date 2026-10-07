"""Aggregate reporting for the five-model transductive comparison.

The paper uses one post-hoc Holm family of 96 hypotheses: four Fusion
contrasts, six pair-depth categories, and AUROC/F1/precision/recall. Corrected
CV intervals are pointwise and do not fully account for network dependence.
No drug records, predictions, models, or training imports are needed here.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from dtpkg.evaluation_stats import (
    comparison_table, holm, normalize_results, performance_summary, stars,
)
from dtpkg.project_paths import EXPERIMENTS_DIR

ORDER = ["low_level-low_level", "low_level-mid_level", "low_level-deep_level",
         "mid_level-mid_level", "mid_level-deep_level", "deep_level-deep_level"]
SHORT = dict(zip(ORDER, ["LL", "LM", "LD", "MM", "MD", "DD"]))
COLORS = {"baseline": "#888888", "fusion": "#0072B2", "topo_only": "#D55E00",
          "common_neighbors": "#009E73", "degree_product": "#CC79A7"}
FIVE_MODELS = ("baseline", "topo_only", "fusion", "common_neighbors", "degree_product")
FUSION_CONTRASTS = tuple(("fusion", model) for model in FIVE_MODELS if model != "fusion")
TRANSDUCTIVE_METRICS = ("auc", "f1", "prec", "rec")
TRANSDUCTIVE_FAMILY = "exploratory_five_model_depth_96"
TRANSDUCTIVE_FAMILY_SIZE = 96
STEM = "transductive_fusion_three_panel"
SUMMARY_FILE = f"{STEM}_summary_ci.csv"
COMPARISON_FILE = f"{STEM}_comparisons_holm.csv"
FOLD_COLUMNS = ["category", "model", "split_repeat", "fold", "training_seed",
                "inference_protocol", "n_outer_train", "n_outer_test",
                *TRANSDUCTIVE_METRICS]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_paths(source_dir=None, out_dir=None):
    experiment = EXPERIMENTS_DIR / "04_transductive_fusion"
    source = experiment / "tables" if source_dir is None else Path(source_dir)
    if (source / "figure_data").is_dir():
        source = source / "figure_data"
    out = experiment / "figures" / "reproduced" if out_dir is None else Path(out_dir)
    return source.resolve(), out.resolve()


def separate_output(source, out):
    """Prevent plotting from overwriting the supplied aggregate evidence."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or source in out.parents:
        raise ValueError("Choose an output directory outside the input tables directory")


def input_paths(source_dir=None):
    source, _ = default_paths(source_dir)
    summary = source / SUMMARY_FILE
    comparisons = source / COMPARISON_FILE
    if not summary.exists():
        summary = source / "main_five_model_lines_summary_ci.csv"
    if not comparisons.exists():
        comparisons = source / "main_contrast_heatmaps_comparisons_holm.csv"
    paths = [summary, comparisons]
    fold = source / "fold_metrics.csv"
    if not fold.exists() and source.name == "figure_data":
        fold = source.parent / "fold_metrics.csv"
    if fold.exists():
        paths.append(fold)
    if (source / "protocol.json").exists():
        paths.append(source / "protocol.json")
    return paths


def _fold_table(frame, protocol=None):
    frame = normalize_results(frame)
    if not set(FOLD_COLUMNS).issubset(frame.columns):
        raise ValueError("Fold scores require model, category, split metadata and all four metrics")
    frame = frame[frame.category.isin(ORDER) & frame.model.isin(FIVE_MODELS)][FOLD_COLUMNS].copy()
    keys = ["category", "model", "split_repeat", "fold"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Require one score per category/model/repeat/fold")
    expected_cells = {(category, model) for category in ORDER for model in FIVE_MODELS}
    if set(frame[["category", "model"]].itertuples(index=False, name=None)) != expected_cells:
        raise ValueError("Missing one of the 30 category/model combinations")
    fold_keys = set(frame[["split_repeat", "fold"]].itertuples(index=False, name=None))
    if protocol is not None:
        expected_folds = {(repeat, fold) for repeat in range(protocol["repeats"])
                          for fold in range(protocol["folds"])}
        if fold_keys != expected_folds:
            raise ValueError("Missing folds from the declared repeated-CV protocol")
    for _, rows in frame.groupby(["category", "model"]):
        if set(rows[["split_repeat", "fold"]].itertuples(index=False, name=None)) != fold_keys:
            raise ValueError("Unmatched folds across models or categories")
    for _, rows in frame.groupby(["split_repeat", "fold"]):
        if any(rows[column].nunique(dropna=False) != 1 for column in
               ("training_seed", "inference_protocol", "n_outer_train", "n_outer_test")):
            raise ValueError("Inconsistent metadata for a matched outer evaluation")
    if set(frame.inference_protocol) != {"cv"}:
        raise ValueError("This reporting procedure requires repeated cross-validation")
    return frame


def calculate_figure_tables(frame, protocol=None):
    """Compute the paper's complete family from aggregate matched fold scores."""
    folds = _fold_table(frame, protocol)
    summary = performance_summary(folds, ("auc", "f1"))
    comparisons = comparison_table(folds, FUSION_CONTRASTS, TRANSDUCTIVE_METRICS,
                                   family=TRANSDUCTIVE_FAMILY, categories=ORDER)
    return summary, comparisons


def export_figure_tables(frame, out_dir, *, repeats=None, folds=None):
    """Write aggregate figure inputs for a new run, without fitting any model."""
    protocol = None
    if repeats is not None or folds is not None:
        if not isinstance(repeats, int) or not isinstance(folds, int) or repeats < 1 or folds < 2:
            raise ValueError("Supply positive repeats and at least two folds together")
        protocol = {"repeats": repeats, "folds": folds}
    scores = _fold_table(frame, protocol)
    summary, comparisons = calculate_figure_tables(scores, protocol)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "fold_metrics.csv", index=False)
    summary.to_csv(out / SUMMARY_FILE, index=False)
    comparisons.to_csv(out / COMPARISON_FILE, index=False)
    if protocol is not None:
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return summary, comparisons


def _compare_numeric(saved, calculated, keys, columns):
    saved = saved.set_index(keys).sort_index()
    calculated = calculated.set_index(keys).sort_index()
    if not saved.index.equals(calculated.index):
        raise ValueError("Saved table identities disagree with the aggregate fold scores")
    np.testing.assert_allclose(saved[columns].to_numpy(float), calculated[columns].to_numpy(float),
                               rtol=0, atol=1e-12, equal_nan=True)


def load_figure_tables(source_dir=None):
    """Return validated `(summary, comparisons)` from public or private tables.

    A private run root or its ``figure_data`` subdirectory is accepted. When
    fold scores are present, every displayed interval and all 96 comparisons
    are recomputed and checked. The public snapshot includes these scores.
    """
    paths = input_paths(source_dir)
    summary, comparisons = (pd.read_csv(path) for path in paths[:2])
    summary = summary[summary.category.isin(ORDER) & summary.model.isin(FIVE_MODELS)
                      & summary.metric.isin(["auc", "f1"])].copy()
    keys = ["category", "metric", "model_a", "model_b"]
    expected = {(cat, metric, "fusion", model) for cat in ORDER
                for metric in TRANSDUCTIVE_METRICS for model in FIVE_MODELS if model != "fusion"}
    if (len(comparisons) != TRANSDUCTIVE_FAMILY_SIZE or comparisons.duplicated(keys).any()
            or set(comparisons[keys].itertuples(index=False, name=None)) != expected
            or not comparisons.family_size.eq(TRANSDUCTIVE_FAMILY_SIZE).all()
            or not comparisons.family.eq(TRANSDUCTIVE_FAMILY).all()):
        raise ValueError("Require all 96 comparisons in the declared transductive family")
    np.testing.assert_allclose(comparisons.p_holm, holm(comparisons.p_raw), rtol=0, atol=1e-12)
    if not comparisons.stars.fillna("").eq(comparisons.p_holm.map(stars)).all():
        raise ValueError("Saved significance markers do not match adjusted p-values")
    expected_summary = {(cat, model, metric) for cat in ORDER for model in FIVE_MODELS
                        for metric in ("auc", "f1")}
    if (len(summary) != 60 or summary.duplicated(["category", "model", "metric"]).any()
            or set(summary[["category", "model", "metric"]].itertuples(index=False, name=None))
            != expected_summary):
        raise ValueError("Require 60 category/model/metric means with their intervals")
    counts = pd.concat([summary.n_folds, comparisons.n_folds])
    if counts.nunique() != 1 or counts.iloc[0] < 2:
        raise ValueError("Every summary and comparison must use the same matched outer evaluations")
    for table in (summary, comparisons):
        if (not table.confidence.eq(.95).all()
                or not table.interval_kind.eq("pointwise_approximate_corrected_cv").all()):
            raise ValueError("The paper figure requires pointwise 95% corrected-CV intervals")
    if not np.isfinite(summary[["estimate", "ci_low", "ci_high"]].to_numpy()).all():
        raise ValueError("The three-panel figure requires finite means and intervals in all categories")
    if not np.isfinite(comparisons.estimate.to_numpy()).all():
        raise ValueError("The three-panel figure requires finite paired effects in all categories")
    if not summary.estimate.between(0, 1).all() or (summary.ci_low > summary.ci_high).any():
        raise ValueError("Invalid saved means or confidence intervals")
    for row in comparisons[comparisons.metric.isin(["auc", "f1"])].itertuples():
        means = summary[(summary.category == row.category) & (summary.metric == row.metric)].set_index("model")
        np.testing.assert_allclose(means.loc["fusion", "estimate"] - means.loc[row.model_b, "estimate"],
                                   row.estimate, rtol=0, atol=1e-12)
    protocol = next((json.loads(path.read_text()) for path in paths if path.name == "protocol.json"), None)
    fold_path = next((path for path in paths if path.name == "fold_metrics.csv"), None)
    if fold_path is not None:
        calculated_summary, calculated_comparisons = calculate_figure_tables(pd.read_csv(fold_path), protocol)
        _compare_numeric(summary, calculated_summary, ["category", "model", "metric"],
                         ["estimate", "ci_low", "ci_high", "se", "n_folds", "df"])
        _compare_numeric(comparisons, calculated_comparisons, keys,
                         ["estimate", "ci_low", "ci_high", "p_raw", "p_holm", "se", "n_folds", "df"])
    return summary, comparisons
