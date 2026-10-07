"""Verify the full 72-contrast family and reproduce the AUROC/F1 heatmap."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dtpkg.evaluation_stats import comparison_table, normalize_results
from dtpkg.mesh_scope.protocol import CATEGORIES, CONTRASTS, METRICS, LEVELS
from dtpkg.mesh_scope.reporting import (
    default_paths, input_hashes, load_protocol, plot_scope_heatmap,
    separate_output, sha256, validate_fold_grid,
)

FIGURE_STEM = "mesh_depth_effect_heatmap"
TABLE_FILES = ("results_pair_bins_per_fold.csv", "scope_comparisons_holm.csv")


def load_comparisons(run_dir=None):
    """Recalculate and verify all 72 paired contrasts from aggregate fold scores.

    Return a dictionary with ``comparisons`` (the verified 72-row DataFrame),
    ``fold_results`` (normalized category scores), ``plan``, ``completion`` and
    ``input_files``. Holm correction always includes all four metrics, even
    though only AUROC and F1 appear in the heatmap.
    """
    run, _ = default_paths(run_dir)
    plan, completion, protocol_files = load_protocol(run)
    n_folds = plan["repeats"] * plan["folds"]
    if (plan.get("inference_family_size") != 72
            or set(plan.get("inference_metrics", [])) != set(METRICS)):
        raise ValueError("Expected the declared 72-hypothesis AUROC/F1/AP/MCC family")
    folds = pd.read_csv(run / "results_pair_bins_per_fold.csv")
    results = normalize_results(folds.rename(columns={"feature_set": "model"}))
    # Private runs also contain overall test scores; the original family contains
    # only the six category rows. Those overall rows play no part in selection.
    results = results[results.category.ne("overall")].copy()
    if set(results.category) != set(CATEGORIES) or set(results.model) != set(LEVELS):
        raise ValueError("Expected all three scopes and all six test categories")
    actual_groups = set(map(tuple, results[["category", "model"]].drop_duplicates().to_numpy()))
    if actual_groups != {(category, model) for category in CATEGORIES for model in LEVELS}:
        raise ValueError("Missing category/scope results")
    validate_fold_grid(results, plan, ["category", "model"])
    calculated = comparison_table(results, CONTRASTS, METRICS,
                                  family="scope_exploratory_72", categories=CATEGORIES)
    saved = pd.read_csv(run / "scope_comparisons_holm.csv")
    keys = ["category", "model_a", "model_b", "metric"]
    if (len(saved) != 72 or saved.duplicated(keys).any()
            or not saved.family_size.eq(72).all()
            or not saved.family.eq("scope_exploratory_72").all()):
        raise ValueError("Expected 72 unique saved contrasts with the original Holm family")
    observed = saved.set_index(keys).sort_index()
    expected = calculated.set_index(keys).sort_index()
    if not observed.index.equals(expected.index) or not observed.n_folds.eq(n_folds).all():
        raise ValueError("Missing or unmatched fold contrasts")
    for column in ("estimate", "se", "ci_low", "ci_high", "p_raw", "p_holm", "confidence", "df"):
        np.testing.assert_allclose(observed[column], expected[column], rtol=0, atol=1e-12, equal_nan=True)
    for column in ("status", "interval_kind", "direction"):
        if not observed[column].equals(expected[column]):
            raise ValueError(f"Saved contrast {column} disagrees with recomputed values")
    if not observed.stars.fillna("").equals(expected.stars.fillna("")):
        raise ValueError("Saved significance labels disagree with the full Holm family")
    return {"comparisons": saved, "fold_results": results,
            "plan": plan, "completion": completion,
            "input_files": protocol_files + TABLE_FILES}


def render_scope_heatmap(run_dir=None, out_dir=None):
    """Return the heatmap PNG; write PNG, caption and provenance to out_dir."""
    run, out = separate_output(*default_paths(run_dir, out_dir))
    result = load_comparisons(run)
    saved, plan = result["comparisons"], result["plan"]
    n_folds = plan["repeats"] * plan["folds"]
    files = result["input_files"]
    hashes = input_hashes(run, files)
    keys = ["category", "model_a", "model_b", "metric"]
    png = plot_scope_heatmap(saved, out, fname=FIGURE_STEM)
    caption_path = out / f"{FIGURE_STEM}_caption.txt"
    caption = (
        "MeSH ontology-scope comparison across test-pair categories. The left and right blocks show "
        "AUROC and F1, respectively. Each cell is the arithmetic mean of paired outer-test score differences "
        f"across {n_folds} matched folds ({plan['repeats']} repeated {plan['folds']}-fold partitions), "
        "calculated as the first named configuration minus the second. Positive values (red) favor "
        "the first configuration; negative values (blue) favor the second. Both metrics share a symmetric "
        "color scale centered at zero. Values are rounded to three decimals; a signed zero can reflect "
        "a small nonzero difference. F1 uses a fixed decision threshold of 0.5. Test scores use the "
        "checkpoint selected on validation AUROC in each fit.\n\n"
        "Rows classify the two drugs by their deepest assigned MeSH level: Low, 1–5; Mid, 6–7; "
        "and Deep, 8–10. Bold row labels identify pairs from the same category. These drug categories "
        "are distinct from the representation configurations: Shallow admits depths 1–5, Intermediate "
        "1–7 and Full 1–10, with inclusive frequency filters of 2–35, 2–63 and 2–52 drugs per tree "
        "position, respectively. The configurations share evaluation pairs, partitions, initial weights "
        "and training-sampling schedules; their filters and independently fitted embeddings differ.\n\n"
        "Asterisks indicate two-sided approximate corrected paired-CV tests with Holm adjustment over "
        "the original 72 hypotheses (six categories × three contrasts × four metrics: AUROC, F1, AP "
        "and MCC), including the two metrics not displayed: * adjusted p < 0.05; ** adjusted p < 0.01. "
        "No asterisk does not establish equivalence; NA would indicate unavailable inference. "
        "These category comparisons are exploratory: folds share drugs and training data, and the "
        "correction does not capture all such dependence. The representation is selected using overall "
        "validation AUROC, separately from this test-category analysis. The comparison does not "
        "isolate ontology depth alone.\n"
    )
    caption_path.write_text(caption)
    if hashes != input_hashes(run, files):
        raise ValueError("Saved inputs changed while rendering the heatmap")
    shown = saved[saved.metric.isin(["auc", "f1"])]
    evidence = dict(
        phase="test", n_matched_folds=n_folds,
        row_order=list(CATEGORIES), metric_order=["auc", "f1"], contrast_order=list(CONTRASTS),
        difference="first configuration minus second; arithmetic mean of paired fold differences",
        holm_family_size=72, displayed_cells=len(shown),
        significant_displayed_cells=shown.loc[shown.p_holm.lt(.05), keys + ["estimate", "p_holm"]].to_dict(orient="records"),
        input_sha256=hashes,
        code_sha256={"renderer": sha256(__file__), "reporting": sha256(Path(__file__).with_name("reporting.py"))},
        artifact_sha256={path.name: sha256(path) for path in (png, caption_path)},
    )
    (out / f"{FIGURE_STEM}_provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Public tables or completed private run directory")
    parser.add_argument("--out", type=Path, help="Output directory (default: figures/reproduced)")
    args = parser.parse_args()
    print(render_scope_heatmap(args.run, args.out))


if __name__ == "__main__":
    main()
