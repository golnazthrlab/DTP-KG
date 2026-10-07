"""Public aggregate reporting for the repeated seen–unseen experiment.

The observation for inference is a seed-averaged drug holdout. The nine fits
are not nine independent observations. Both Holm families retain unplotted
metrics. The overlap correction is exploratory, with three holdouts (df=2).
No private drug records or training dependencies are required here.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from dtpkg.evaluation_stats import corrected_interval, holm
from dtpkg.project_paths import EXPERIMENTS_DIR

ARM = "biological_only"
SCENARIO = "seen_unseen"
SPLIT_IDS = ("split_00", "split_01", "split_02")
SEEDS = (101, 202, 303)
MODELS = ("baseline", "topo_only", "fusion")
DEPTH = ("low_level-low_level", "low_level-mid_level", "low_level-deep_level",
         "mid_level-mid_level", "mid_level-deep_level", "deep_level-deep_level")
CELLS = (("overall", "overall"), *[("depth", category) for category in DEPTH])
CELL_KEYS = ["scenario", "group_kind", "category"]
SUMMARY_METRICS = ("auc", "f1", "prec", "rec", "acc", "ap")
N_HELDOUT, N_DEVELOPMENT = 374, 2119
INFERENCE_SPEC = dict(
    overall=dict(family="seen_unseen_overall_primary", family_size=4,
                 metrics=["auc", "f1"], categories=["overall"]),
    depth=dict(family="seen_unseen_depth_exploratory", family_size=60,
               metrics=["auc", "f1", "prec", "rec", "acc"], categories=list(DEPTH)),
    contrasts=[["fusion", "baseline"], ["fusion", "topo_only"]],
    unit="training-seed-averaged drug holdout", n_holdouts=3, df=2,
    correction_ratio=N_HELDOUT / N_DEVELOPMENT,
    interpretation="Approximate overlap correction; not an exact test for shared-drug/network dependence",
)
METRIC_COLUMNS = [*CELL_KEYS, "analysis_arm", "split_id", "split_repeat", "training_seed", "model",
                  "n_heldout_drugs", "n_development_drugs", "n", "n_pos", "n_neg", *SUMMARY_METRICS]
TABLE_FILES = {
    "metrics": "seen_unseen_per_split_seed_metrics.csv",
    "summary": "seen_unseen_summary.csv",
    "overall": "seen_unseen_overall_comparisons.csv",
    "depth": "seen_unseen_depth_pointwise_comparisons.csv",
}

def _complete_mean(values):
    values = np.asarray(values, float)
    return float(values.mean()) if np.isfinite(values).all() else np.nan

def _complete_sd(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1)) if np.isfinite(values).all() else np.nan

def summarize_metrics(metrics):
    """Validate the 3×3×3 grid, then average seeds before comparing holdouts."""
    required = {*CELL_KEYS, "analysis_arm", "split_id", "split_repeat", "training_seed", "model",
                "n_heldout_drugs", "n_development_drugs", "n", "n_pos", "n_neg", *SUMMARY_METRICS}
    if not required <= set(metrics):
        raise ValueError(f"Missing metric columns: {sorted(required - set(metrics))}")
    if set(metrics.analysis_arm) != {ARM} or set(metrics.scenario) != {SCENARIO}:
        raise ValueError("Expected only biological_only seen–unseen metrics")
    scores = metrics[list(SUMMARY_METRICS)].to_numpy(float)
    if np.isinf(scores).any() or ((scores[np.isfinite(scores)] < 0) | (scores[np.isfinite(scores)] > 1)).any():
        raise ValueError("Metric values must be within [0, 1] or undefined")
    keys = [*CELL_KEYS, "split_id", "training_seed", "model"]
    expected = {(SCENARIO, kind, category, split, seed, model)
                for kind, category in CELLS for split in SPLIT_IDS for seed in SEEDS for model in MODELS}
    if (metrics[keys].isna().any().any() or len(metrics) != len(expected) or
            set(metrics[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError("Require the complete unique three-holdout, three-seed, three-model metric grid")
    if (not metrics.n_heldout_drugs.eq(N_HELDOUT).all() or
            not metrics.n_development_drugs.eq(N_DEVELOPMENT).all()):
        raise ValueError("Repeated analysis requires 374 held-out and 2119 development drugs per split")
    for split_index, split in enumerate(SPLIT_IDS):
        if not metrics.loc[metrics.split_id == split, "split_repeat"].eq(split_index).all():
            raise ValueError("Split ID and repeat index disagree")
    for _, cell in metrics.groupby([*CELL_KEYS, "split_id"]):
        if any(cell[column].nunique(dropna=False) != 1 for column in ("n", "n_pos", "n_neg")):
            raise ValueError("Models and seeds must use identical pair support within each holdout")
        if (not cell.n.eq(cell.n_pos + cell.n_neg).all() or
                not np.isfinite(cell[["n", "n_pos", "n_neg"]].to_numpy(float)).all() or
                not cell[["n", "n_pos", "n_neg"]].ge(0).all().all() or
                not cell[["n", "n_pos", "n_neg"]].mod(1).eq(0).all().all()):
            raise ValueError("Invalid pair support counts")
    holdouts = []
    for key, group in metrics.groupby([*CELL_KEYS, "split_id", "model"], sort=True):
        metadata = dict(zip([*CELL_KEYS, "split_id", "model"], key), analysis_arm=ARM,
                        split_repeat=int(group.split_repeat.iloc[0]),
                        n_training_seeds=len(SEEDS), n_pairs=int(group.n.iloc[0]),
                        n_pos=int(group.n_pos.iloc[0]), n_neg=int(group.n_neg.iloc[0]),
                        n_heldout_drugs=N_HELDOUT, n_development_drugs=N_DEVELOPMENT)
        group = group.set_index("training_seed").reindex(SEEDS)
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            holdouts.append(dict(metadata, metric=metric, estimate=_complete_mean(values),
                                 seed_sd=_complete_sd(values), n_supported_seeds=int(np.isfinite(values).sum()),
                                 **{f"seed_{seed}": value for seed, value in zip(SEEDS, values)}))
    holdouts = pd.DataFrame(holdouts)
    summary, comparisons = [], []
    for key, group in holdouts.groupby([*CELL_KEYS, "metric"], sort=True):
        scenario, kind, category, metric = key
        metadata = dict(analysis_arm=ARM, scenario=scenario, group_kind=kind,
                        category=category, metric=metric, n_holdouts=len(SPLIT_IDS), n_training_seeds=len(SEEDS))
        values_by_model = {}
        for model in MODELS:
            rows = group[group.model == model].set_index("split_id").reindex(SPLIT_IDS)
            values = rows.estimate.to_numpy(float)
            values_by_model[model] = values
            summary.append(dict(metadata, model=model, estimate=_complete_mean(values),
                sd_across_holdouts=_complete_sd(values),
                mean_within_holdout_seed_sd=_complete_mean(rows.seed_sd),
                n_supported_holdouts=int(np.isfinite(values).sum()),
                interval_kind="between_holdout_sample_sd", aggregation="mean_of_seed_averaged_holdouts",
                **{split: value for split, value in zip(SPLIT_IDS, values)}))
        spec = INFERENCE_SPEC[kind]
        if metric not in spec["metrics"]:
            continue
        for comparator in ("baseline", "topo_only"):
            differences = values_by_model["fusion"] - values_by_model[comparator]
            inference = corrected_interval(differences, np.full(len(SPLIT_IDS), N_HELDOUT / N_DEVELOPMENT))
            inference.update(estimate=_complete_mean(differences),
                             interval_kind="exploratory_corrected_drug_holdout")
            comparisons.append(dict(metadata, model_a="fusion", model_b=comparator,
                family=spec["family"], family_size=spec["family_size"],
                sd_across_holdouts=_complete_sd(differences),
                n_supported_holdouts=int(np.isfinite(differences).sum()),
                **{split: value for split, value in zip(SPLIT_IDS, differences)}, **inference))
    summary, comparisons = pd.DataFrame(summary), pd.DataFrame(comparisons)
    for kind in ("overall", "depth"):
        spec = INFERENCE_SPEC[kind]
        index = comparisons.index[comparisons.family == spec["family"]]
        if len(index) != spec["family_size"]:
            raise ValueError("Comparison family is incomplete")
        comparisons.loc[index, "p_holm"] = holm(comparisons.loc[index, "p_raw"])
    return holdouts, summary, comparisons


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_paths(source=None, out_dir=None):
    experiment = EXPERIMENTS_DIR / "05_inductive_fusion"
    source = experiment / "tables" if source is None else Path(source)
    output = experiment / "figures" / "reproduced" if out_dir is None else Path(out_dir)
    return source.resolve(), output.resolve()


def separate_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Choose an output directory outside the input tables directory")


def depth_comparisons(pointwise):
    """Apply Holm across all 60 tests, including the metrics not plotted."""
    keys = ["category", "metric", "model_a", "model_b"]
    required = {"scenario", "group_kind", "p_raw", "p_pointwise", "estimate", "sd_across_holdouts", *keys}
    if not required <= set(pointwise):
        raise ValueError(f"Missing pointwise columns: {sorted(required - set(pointwise))}")
    expected = {(category, metric, "fusion", comparator) for category in DEPTH
                for metric in INFERENCE_SPEC["depth"]["metrics"] for comparator in ("baseline", "topo_only")}
    if (len(pointwise) != 60 or pointwise[keys].isna().any().any() or
            set(pointwise[keys].itertuples(index=False, name=None)) != expected or
            set(pointwise.scenario) != {SCENARIO} or set(pointwise.group_kind) != {"depth"}):
        raise ValueError("Holm depth comparisons require the complete 60-test family")
    if not np.allclose(pointwise.p_pointwise, pointwise.p_raw, rtol=0, atol=0, equal_nan=True):
        raise ValueError("Pointwise p-values must equal the raw corrected-test p-values")
    frame = pointwise.copy(deep=True)
    frame["p_holm"] = holm(frame.p_raw)
    frame["family"] = "seen_unseen_depth_exploratory"
    frame["family_size"] = 60
    frame["multiplicity_adjustment"] = "Holm"
    return frame


def _check_saved(saved, recomputed, keys, columns, label):
    if not set(keys + columns) <= set(saved):
        raise ValueError(f"Incomplete {label} columns")
    if saved.duplicated(keys).any() or saved[keys].isna().any().any():
        raise ValueError(f"Duplicate or missing {label} cell")
    left = saved.set_index(keys).sort_index()
    right = recomputed.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError(f"Incomplete {label} grid")
    if not np.allclose(left[columns].to_numpy(float), right[columns].to_numpy(float),
                       atol=1e-12, rtol=1e-10, equal_nan=True):
        raise ValueError(f"Saved {label} disagrees with the seed-averaged holdout scores")


def load_report(source=None):
    """Validate public aggregate scores, summaries, and complete test families.

    ``source`` may also be a private aggregate export produced by write_report.
    It must contain the four declared CSV files and snapshot_provenance.json.
    Input files are never modified.
    """
    source, _ = default_paths(source)
    manifest_path = source / "snapshot_provenance.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported aggregate report schema")
    recorded = manifest.get("files", {})
    inputs = {manifest_path: sha256(manifest_path)}
    if "protocol.json" in recorded:
        protocol = source / "protocol.json"
        if sha256(protocol) != recorded["protocol.json"]:
            raise ValueError("Protocol does not match recorded provenance")
        inputs[protocol] = recorded["protocol.json"]
    tables = {}
    for key, name in TABLE_FILES.items():
        path = source / name
        digest = sha256(path)
        if recorded.get(name) != digest:
            raise ValueError(f"Aggregate table does not match recorded provenance: {name}")
        inputs[path] = digest
        tables[key] = pd.read_csv(path)
    holdouts, summary, comparisons = summarize_metrics(tables["metrics"])
    _check_saved(tables["summary"], summary, [*CELL_KEYS, "metric", "model"],
                 ["estimate", "sd_across_holdouts", "mean_within_holdout_seed_sd",
                  "n_supported_holdouts", "n_holdouts", "n_training_seeds", *SPLIT_IDS], "summary")
    comparison_keys = [*CELL_KEYS, "metric", "model_a", "model_b"]
    numeric = ["estimate", "sd_across_holdouts", "p_raw", "se", "ci_low", "ci_high", "df",
               "n_folds", "n_holdouts", "n_training_seeds", "n_supported_holdouts", *SPLIT_IDS]
    _check_saved(tables["overall"], comparisons[comparisons.group_kind.eq("overall")],
                 comparison_keys, [*numeric, "p_holm", "family_size"], "overall comparisons")
    depth = depth_comparisons(tables["depth"])
    _check_saved(depth, comparisons[comparisons.group_kind.eq("depth")], comparison_keys,
                 [*numeric, "p_holm", "family_size"], "depth comparisons")
    tables["depth"] = depth
    tables["holdouts"] = holdouts
    tables["comparisons"] = comparisons
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed while loading: {path.name}")
    return tables


def write_report(metrics, output):
    """Export only aggregate results from a complete seen–unseen run.

    Select the three neural models and overall/depth rows before calling this
    function. A new output directory is required to protect previous results.
    """
    _, summary, comparisons = summarize_metrics(metrics)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    overall = comparisons[comparisons.group_kind.eq("overall")].copy()
    depth = comparisons[comparisons.group_kind.eq("depth")].drop(columns=["p_holm"]).copy()
    depth["p_pointwise"] = depth.p_raw
    tables = dict(metrics=metrics[METRIC_COLUMNS], summary=summary, overall=overall, depth=depth)
    for key, table in tables.items():
        table.to_csv(output / TABLE_FILES[key], index=False)
    manifest = dict(schema_version=1, content="aggregate seen-unseen scores without drug or pair records",
                    inference=INFERENCE_SPEC,
                    files={name: sha256(output / name) for name in TABLE_FILES.values()})
    (output / "snapshot_provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_report(output)
    return output
