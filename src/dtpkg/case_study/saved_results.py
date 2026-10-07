"""Load private historical results after checking every retained artifact."""
import json
from pathlib import Path

import pandas as pd

from .reporting import SETTINGS, verify_snapshot


TABLES = ("pair_overlap_scores", "correlations_by_fit", "correlations_by_split",
          "correlation_summary", "selected_examples", "example_selection", "coverage", "paper_table")


def load_saved_results(setting, results_root):
    """Return historical private tables without claiming current-model replay."""
    if setting not in SETTINGS:
        raise ValueError("Choose transductive or inductive")
    root = Path(results_root).expanduser().resolve()
    snapshot = verify_snapshot(root)
    required = [f"{setting}/{name}.csv" for name in TABLES]
    required.append(f"{setting}/source_provenance.json")
    if not set(required) <= set(snapshot["retained_files"]):
        raise ValueError("Private inputs are missing from the retained inventory")
    manifest = json.loads((root / snapshot["source_run_manifests"][setting]).read_text())
    tables = {name: pd.read_csv(root / setting / f"{name}.csv") for name in TABLES}
    return {**tables, "folder": root / setting, "manifest": manifest,
            "source_provenance": json.loads((root / setting / "source_provenance.json").read_text()),
            "snapshot_manifest": snapshot, "validation_scope": "retained_artifacts_only",
            "no_current_model_source_verification": True}
