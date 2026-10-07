"""Export verified case-study aggregates without pair or graph records."""
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd


GROUPS = ["setting", "scenario", "subset", "label_group", "feature",
          "statistic", "controls", "control_encoding"]
FIT_COLUMNS = ["setting", "split_id", "split_repeat", "training_seed", "fold", "fit_id",
               "scenario", "subset", "label_group", "feature", "rho", "statistic", "controls",
               "control_encoding", "n", "n_total", "n_undefined_feature", "n_unique_pairs",
               "n_drugs", "n_unique_scores", "n_zero_overlap", "n_saturated_scores",
               "n_total_zero_overlap", "n_total_saturated_scores", "status", "design_rank", "residual_df"]
SPLIT_COLUMNS = [*GROUPS, "split_repeat", "mean_rho", "n_fits", "n_min", "n_max", "n_defined_fits", "status"]
SUMMARY_COLUMNS = [*GROUPS, "mean_rho", "sd_rho", "min_rho", "max_rho", "n_splits",
                   "n_defined_splits", "n_fits", "status", "n_min", "n_max", "interpretation"]
COVERAGE_COLUMNS = ["setting", "scenario", "n_prediction_rows", "n_unique_pairs", "n_unique_drugs",
                    "n_fits", "n_positive_rows", "n_negative_rows", "n_zero_overlap_rows", "n_both_target_rows"]
SCHEMAS = {"correlations_by_fit": FIT_COLUMNS, "correlations_by_split": SPLIT_COLUMNS,
           "correlation_summary": SUMMARY_COLUMNS, "coverage": COVERAGE_COLUMNS,
           "adjusted_splits": SPLIT_COLUMNS, "adjusted_summary": SUMMARY_COLUMNS}
SETTINGS = ("transductive", "inductive")
SERIES = (
    ("transductive", "shared_protein_count"),
    ("transductive", "biological_shared_protein_count"),
    ("seen_unseen", "biological_shared_protein_count"),
    ("unseen_unseen", "biological_shared_protein_count"),
)
ALLOWED_TEXT = {
    "setting": set(SETTINGS),
    "scenario": {"transductive", "seen_unseen", "unseen_unseen"},
    "subset": {"all_pairs", "both_have_targets"},
    "label_group": {"overall", "positive", "negative"},
    "feature": {"shared_protein_count", "biological_shared_protein_count", "shared_target_count",
                "protein_jaccard", "protein_size_geomean"},
    "statistic": {"spearman", "partial_spearman"},
    "controls": {"none", "n_proteins_a|n_proteins_b", "n_targets_a|n_targets_b",
                 "n_biological_proteins_a|n_biological_proteins_b",
                 "n_biological_proteins_a|n_biological_proteins_b|either_biological_size_zero"},
    "control_encoding": {"none", "endpoint_columns", "unordered_minmax",
                         "endpoint_columns_plus_zero", "unordered_minmax_plus_zero"},
    "status": {"ok", "insufficient_pairs", "constant_feature", "constant_score",
               "insufficient_residual_df", "constant_feature_residual", "constant_score_residual",
               "undefined_fit", "undefined_split"},
    "interpretation": {"descriptive; SD and range are not confidence intervals"},
}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid artifact path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name == ".":
        raise ValueError("Invalid artifact path")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes results directory")
    return path


def _check_inventory(root, inventory):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("A nonempty artifact inventory is required")
    for name, expected in inventory.items():
        path = _relative_path(root, name)
        if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
            raise ValueError(f"Invalid SHA-256 digest: {name}")
        if not path.is_file() or _hash(path) != expected.lower():
            raise ValueError(f"Artifact hash mismatch: {name}")


def verify_snapshot(results_root):
    """Verify every retained historical artifact; do not recheck current models."""
    root = Path(results_root).expanduser().resolve()
    snapshot = json.loads((root / "snapshot_manifest.json").read_text())
    if snapshot.get("format_version") != 1 or snapshot.get("kind") != "case_study_saved_results":
        raise ValueError("Unsupported case-study snapshot")
    date.fromisoformat(snapshot["snapshot_date"])
    inventory = snapshot.get("retained_files")
    _check_inventory(root, inventory)
    expected = {setting: f"{setting}/original_analysis_manifest.json" for setting in SETTINGS}
    if snapshot.get("source_run_manifests") != expected or not set(expected.values()) <= set(inventory):
        raise ValueError("Both original analysis manifests must be retained")
    for setting, name in expected.items():
        manifest = json.loads((root / name).read_text())
        if manifest.get("setting") != setting or manifest.get("complete") is not True:
            raise ValueError("Incomplete or mismatched original analysis")
    return snapshot


def _table(frame, columns, name):
    """Explicit allowlist: unknown fields cannot enter public exports."""
    if list(frame.columns) != columns or frame.empty:
        raise ValueError(f"Unexpected or empty aggregate schema: {name}")
    frame = frame.copy()
    for column in columns:
        values = frame[column]
        if column in ALLOWED_TEXT:
            if not set(values) <= ALLOWED_TEXT[column]:
                raise ValueError(f"Unexpected aggregate values: {name}/{column}")
        elif column in ("split_id", "fit_id"):
            pattern = (r"(?:repeat|split)_\d+" if column == "split_id" else
                       r"(?:transductive_repeat_\d+_fold_\d+|inductive_split_\d+_seed_\d+)")
            if not values.astype(str).str.fullmatch(pattern).all():
                raise ValueError(f"Invalid aggregate fit identifier: {name}/{column}")
        else:
            numbers = pd.to_numeric(values, errors="raise")
            if np.isinf(numbers.to_numpy(float)).any():
                raise ValueError(f"Infinite aggregate value: {name}/{column}")
            if column not in {"rho", "mean_rho", "sd_rho", "min_rho", "max_rho", "design_rank", "residual_df"}:
                if numbers.isna().any() or (numbers < 0).any() or (numbers != np.floor(numbers)).any():
                    raise ValueError(f"Invalid aggregate count: {name}/{column}")
            if column in {"rho", "mean_rho", "min_rho", "max_rho"} and (numbers.abs() > 1 + 1e-12).any():
                raise ValueError(f"Invalid correlation: {name}/{column}")
            frame[column] = numbers
    return frame


def _same(left, right, keys, name):
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    except AssertionError as error:
        raise ValueError(f"Aggregate estimates disagree: {name}") from error


def _selected(frame):
    common = (frame.subset.eq("all_pairs") & frame.label_group.eq("overall")
              & frame.statistic.eq("partial_spearman") & frame.control_encoding.eq("unordered_minmax"))
    return pd.concat([frame.loc[common & frame.scenario.eq(scenario) & frame.feature.eq(feature)]
                      for scenario, feature in SERIES], ignore_index=True)


def _validate_tables(tables):
    from .analysis import summarize_correlations

    tables = {name: _table(tables[name], columns, name) for name, columns in SCHEMAS.items()}
    fits = tables["correlations_by_fit"]
    if fits.duplicated([*GROUPS, "fit_id"]).any():
        raise ValueError("Duplicate aggregate fit")
    coverage = tables["coverage"]
    if coverage.duplicated(["setting", "scenario"]).any() or len(coverage) != 3:
        raise ValueError("Incomplete aggregate coverage")
    overall = fits[fits.subset.eq("all_pairs") & fits.label_group.eq("overall")]
    for (setting, scenario), rows in overall.groupby(["setting", "scenario"]):
        saved = coverage[coverage.setting.eq(setting) & coverage.scenario.eq(scenario)]
        totals = rows.groupby("fit_id").n_total
        if (len(saved) != 1 or (totals.nunique() != 1).any()
                or saved.iloc[0].n_fits != rows.fit_id.nunique()
                or saved.iloc[0].n_prediction_rows != totals.first().sum()
                or saved.iloc[0].n_positive_rows + saved.iloc[0].n_negative_rows != saved.iloc[0].n_prediction_rows):
            raise ValueError("Aggregate coverage disagrees with fit statistics")
    splits, summary = summarize_correlations(fits)
    _same(tables["correlations_by_split"], splits[SPLIT_COLUMNS], [*GROUPS, "split_repeat"], "split means")
    _same(tables["correlation_summary"], summary[SUMMARY_COLUMNS], GROUPS, "summary")
    for name, original in (("adjusted_splits", "correlations_by_split"), ("adjusted_summary", "correlation_summary")):
        _same(tables[name], _selected(tables[original]), [*GROUPS, *(["split_repeat"] if name.endswith("splits") else [])], name)
    for scenario, feature in SERIES:
        values = tables["adjusted_splits"].query("scenario == @scenario and feature == @feature")
        summary = tables["adjusted_summary"].query("scenario == @scenario and feature == @feature")
        if (len(values) < 2 or len(summary) != 1 or not values.status.eq("ok").all()
                or not summary.status.eq("ok").all()
                or set(values.split_repeat) != set(range(len(values)))):
            raise ValueError(f"Incomplete adjusted series: {scenario}/{feature}")
        setting = "transductive" if scenario == "transductive" else "inductive"
        expected = set(fits.loc[fits.setting.eq(setting), "split_repeat"])
        if set(values.split_repeat) != expected:
            raise ValueError(f"Missing adjusted split: {scenario}/{feature}")
    return tables


def _protocol(tables, snapshot_date):
    return {
        "format_version": 1, "study": "case_study", "snapshot_date": snapshot_date,
        "model": "topo_only", "radius_from_drug": 2,
        "transductive_condition": "full graph; biological-only overlap sensitivity uses the same predictions",
        "inductive_condition": "DDI-free graph for training, inference, and overlap",
        "split_repeats": {setting: sorted(int(x) for x in tables["correlations_by_fit"].loc[
            tables["correlations_by_fit"].setting.eq(setting), "split_repeat"].unique()) for setting in SETTINGS},
        "training_seeds": {setting: sorted(int(x) for x in tables["correlations_by_fit"].loc[
            tables["correlations_by_fit"].setting.eq(setting), "training_seed"].unique()) for setting in SETTINGS},
        "aggregation": "equal-weight split means after averaging training seeds within folds and then folds",
        "adjustment": "OLS residuals of average-tied ranks, intercept and ranked smaller/larger neighborhood sizes; residuals correlated without reranking",
        "interpretation": "exploratory; descriptive variation, not confidence intervals",
    }


def load_tables(table_dir):
    """Load public aggregates and verify hashes, schemas and all summary values."""
    root = Path(table_dir).expanduser().resolve()
    provenance = json.loads((root / "provenance.json").read_text())
    names = {f"{name}.csv" for name in SCHEMAS} | {"protocol.json"}
    if provenance.get("format_version") != 1 or set(provenance.get("public_files", {})) != names:
        raise ValueError("Incomplete public aggregate inventory")
    _check_inventory(root, provenance["public_files"])
    tables = _validate_tables({name: pd.read_csv(root / f"{name}.csv") for name in SCHEMAS})
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol != _protocol(tables, protocol.get("snapshot_date")):
        raise ValueError("Aggregate protocol disagrees with saved tables")
    return {**tables, "protocol": protocol, "provenance": provenance}


def export_tables(results_root, output_dir):
    """Export aggregates from a verified snapshot or a verified fresh analysis.

    Snapshot verification covers all retained bytes, including private artifacts,
    but only the allowlisted aggregate columns are written to the public folder.
    Fresh analyses are verified against their current source and output hashes.
    """
    root = Path(results_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    if (out == root or root.is_relative_to(out)
            or out in {root / name for name in (*SETTINGS, "comparison")}):
        raise ValueError("Choose a separate aggregate output directory")
    names = tuple(name for name in SCHEMAS if not name.startswith("adjusted_"))
    source_files = {}
    if (root / "snapshot_manifest.json").exists():
        snapshot = verify_snapshot(root)
        source = {}
        for setting in SETTINGS:
            source[setting] = {}
            for name in names:
                relative = f"{setting}/{name}.csv"
                if relative not in snapshot["retained_files"]:
                    raise ValueError(f"Aggregate input is not retained: {relative}")
                source[setting][name] = pd.read_csv(root / relative)
                source_files[relative] = _hash(root / relative)
        provenance = {"format_version": 1, "validation_scope": "retained_artifacts_only",
                      "no_current_model_source_verification": True,
                      "snapshot_manifest_sha256": _hash(root / "snapshot_manifest.json"),
                      "verified_retained_files": len(snapshot["retained_files"])}
        snapshot_date = snapshot["snapshot_date"]
    else:
        from .workflow import load_results
        source = {setting: load_results(setting, results_root=root) for setting in SETTINGS}
        source_files = {f"{setting}/{name}.csv": _hash(root / setting / f"{name}.csv")
                        for setting in SETTINGS for name in names}
        provenance = {"format_version": 1, "validation_scope": "current_analysis_sources_and_outputs",
                      "no_current_model_source_verification": False}
        snapshot_date = None
    tables = {name: pd.concat([source[setting][name] for setting in SETTINGS], ignore_index=True)
              for name in names}
    tables["adjusted_splits"] = _selected(tables["correlations_by_split"])
    tables["adjusted_summary"] = _selected(tables["correlation_summary"])
    tables = _validate_tables(tables)
    if snapshot_date is not None:
        relative = "comparison/writing_table.csv"
        if relative not in snapshot["retained_files"]:
            raise ValueError("Writing-table input is not retained")
        writing = pd.read_csv(root / relative)
        _same(tables["adjusted_summary"], _selected(writing)[SUMMARY_COLUMNS], GROUPS, "historical figure summary")
        source_files[relative] = _hash(root / relative)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(_protocol(tables, snapshot_date), indent=2) + "\n")
    provenance["source_aggregate_files"] = source_files
    provenance["public_files"] = {name: _hash(out / name) for name in sorted(
        {f"{key}.csv" for key in tables} | {"protocol.json"})}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return load_tables(out)
