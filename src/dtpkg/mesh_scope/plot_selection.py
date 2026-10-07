"""Validate saved validation scores and reproduce the MeSH selection figure.

Defaults to the public aggregate tables. This module never fits a model or reads
outer-test scores to choose the representation. All outputs go to ``out_dir``.
"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from dtpkg.mesh_scopes import SCOPES
from dtpkg.mesh_scope.reporting import (
    default_paths, input_hashes, load_protocol, separate_output, sha256,
    validate_fold_grid,
)

FIGURE_STEM = "mesh_scope_validation_selection"
TABLE_FILES = ("training_log.csv", "validation_metrics.csv",
               "validation_scope_selection.csv")


def load_selection(run_dir=None):
    """Return validated fold scores and the validation-only scope choice.

    The dictionary contains ``fold_values`` (wide DataFrame indexed by repeat
    and fold), ``means`` (Series), ``selected`` (scope key), ``plan`` and
    ``completion`` (protocol dictionaries), and ``input_files`` (filenames).
    No test table is opened. Exact ties prefer the shallower configuration.
    """
    run, _ = default_paths(run_dir)
    plan, completion, protocol_files = load_protocol(run)
    levels = [s.level_key for s in SCOPES]
    logs = pd.read_csv(run / "training_log.csv")
    metrics = pd.read_csv(run / "validation_metrics.csv")
    selection = pd.read_csv(run / "validation_scope_selection.csv")
    for table in (logs, selection):
        if set(table.feature_set) != set(levels):
            raise ValueError("Expected exactly the three MeSH configurations")
    if selection.feature_set.duplicated().any() or len(selection) != len(levels):
        raise ValueError("Expected one selection summary per configuration")
    saved = selection.set_index("feature_set").loc[levels]
    overall = metrics[metrics.pair_bin.eq("overall")]
    if set(overall.feature_set) != set(levels):
        raise ValueError("Missing overall validation scores")
    validate_fold_grid(logs, plan, ["feature_set"])
    validate_fold_grid(overall, plan, ["feature_set"])
    wide = logs.pivot(index=["split_repeat", "fold"], columns="feature_set",
                      values="best_val_auc").loc[:, levels].sort_index()
    restored = overall.pivot(index=["split_repeat", "fold"], columns="feature_set",
                             values="auc").loc[:, levels].sort_index()
    if (wide.empty or not np.isfinite(wide.to_numpy()).all()
            or not wide.index.equals(restored.index)):
        raise ValueError("Missing or unmatched scope validation results")
    np.testing.assert_allclose(wide, restored, rtol=0, atol=1e-12)
    if not wide.ge(0).all().all() or not wide.le(1).all().all():
        raise ValueError("Validation AUROC must lie between zero and one")
    means = wide.mean()
    np.testing.assert_allclose(means, saved["mean"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(saved["count"].to_numpy(),
                                  np.repeat(len(wide), len(levels)))
    winner = means.idxmax()
    if not saved.preferred_by_validation_auc.isin([True, False]).all():
        raise ValueError("Invalid saved validation-selection flags")
    if saved.index[saved.preferred_by_validation_auc.astype(bool)].tolist() != [winner]:
        raise ValueError("Saved selection disagrees with mean validation AUROC")
    return {"fold_values": wide, "means": means, "selected": winner,
            "plan": plan, "completion": completion,
            "input_files": protocol_files + TABLE_FILES}


def plot_scope_selection(run_dir=None, out_dir=None):
    """Return a PNG after validating scores; write all artifacts to out_dir.

    Existing reference figures and input tables are unchanged by the default
    call. The output directory also receives the caption, fold scores and
    provenance. Inputs may be the public snapshot or a completed private run.
    """
    run, out = separate_output(*default_paths(run_dir, out_dir))
    result = load_selection(run)
    files = result["input_files"]
    hashes = input_hashes(run, files)
    wide, means, winner = result["fold_values"], result["means"], result["selected"]
    plan = result["plan"]
    levels = [s.level_key for s in SCOPES]
    repeats, folds = plan["repeats"], plan["folds"]
    scope_names = [s.name for s in SCOPES]
    selected = scope_names[levels.index(winner)]
    wins = int(wide.idxmax(axis=1).eq(winner).sum())
    out.mkdir(parents=True, exist_ok=True)
    stem = out / FIGURE_STEM
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        fig.subplots_adjust(left=.13, right=.98, bottom=.20, top=.98)
        x = np.arange(len(levels))
        offsets = np.linspace(-.055, .055, len(wide))
        for offset, row in zip(offsets, wide.to_numpy()):
            ax.plot(x + offset, row, color="#aab2bb", alpha=.42, lw=.8, zorder=1)
        for i, scope in enumerate(SCOPES):
            ax.scatter(i + offsets, wide[scope.level_key], s=23, color=scope.color,
                       alpha=.85, edgecolors="white", linewidths=.35, zorder=2)
            ax.scatter([i], [means[scope.level_key]], marker="D", s=110,
                       color=scope.color, edgecolors="#18212b", linewidths=1.1, zorder=4)
        ax.set_xticks(x, scope_names)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_xlim(-.35, len(levels)-.65)
        lo, hi = float(wide.min().min()), float(wide.max().max())
        ax.set_ylim(max(0, lo-.003), min(1, hi+.003))
        ax.yaxis.set_major_locator(MultipleLocator(.01))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.set_ylabel("Validation AUROC")
        ax.grid(axis="y", color="#e4e8ec", lw=.8)
        ax.set_axisbelow(True)
        handles = [Line2D([0], [0], color="#aab2bb", marker="o", markersize=4, lw=.8,
                          label="Matched fold results"),
                   Line2D([0], [0], color="#18212b", marker="D", markersize=7, lw=0,
                          label="Mean validation AUROC")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55,.01),
                   ncol=2, frameon=False, fontsize=9)
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    caption = (
        f"Selection of the primary MeSH representation using overall validation AUROC. Small points show "
        f"overall validation AUROC at the restored selected checkpoint for each of {len(wide)} matched folds "
        f"from {repeats} repeated {folds}-fold partitions; lines connect configurations evaluated on the same split. "
        "Small horizontal offsets separate overlapping points and have no quantitative meaning. "
        "Within each fold, AUROC is calculated over all validation pairs together, including comparisons "
        "between positive and negative pairs from different interaction categories; it is not the mean "
        "of the six category-specific AUROCs. Large diamonds indicate the arithmetic mean of these "
        "fold-level overall AUROCs: " + ", ".join(f"{s.name} {means[s.level_key]:.4f}" for s in SCOPES) + ". "
        f"{selected} had the highest mean and was selected for subsequent experiments; it ranked first in "
        f"{wins}/{len(wide)} matched folds. Shallow, Intermediate and Full retain tree positions at depths "
        "1–5, 1–7 and 1–10, with inclusive frequency ranges of 2–35, 2–63 and 2–52 drugs per position, "
        "respectively. All configurations use identical evaluation pairs and matched partitions; "
        f"{100 * plan['validation_fraction']:g}% of each outer development partition is reserved for validation. These are validation "
        "selection scores, not outer-test performance estimates or category-specific results. "
        "Scores are correlated and were used for checkpoint/configuration "
        "selection; this descriptive plot does not provide an unbiased confirmatory performance estimate. "
        "Configurations use their prescribed frequency filters and independently fitted embeddings, so "
        "their comparison does not isolate ontology depth alone.\n"
    )
    caption_path = stem.with_name(stem.name + "_caption.txt")
    caption_path.write_text(caption)
    fold_path = out / f"{FIGURE_STEM}_fold_values.csv"
    wide.reset_index().to_csv(fold_path, index=False)
    if hashes != input_hashes(run, files):
        raise ValueError("Saved result inputs changed while plotting")
    evidence = dict(checkpoint_metric="validation AUROC",
                    means=means.to_dict(), selected=winner, n_matched_folds=len(wide),
                    selected_fold_wins=wins, repeats=repeats, folds=folds,
                    inference="descriptive; no tests or confidence intervals",
                    input_sha256=hashes, renderer_sha256=sha256(__file__),
                    artifact_sha256={path.name: sha256(path)
                                     for path in (stem.with_suffix(".png"), caption_path, fold_path)})
    (out / f"{FIGURE_STEM}_provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Public tables or completed private run directory")
    parser.add_argument("--out", type=Path, help="Output directory (default: figures/reproduced)")
    args = parser.parse_args()
    print(plot_scope_selection(args.run, args.out))


if __name__ == "__main__":
    main()
