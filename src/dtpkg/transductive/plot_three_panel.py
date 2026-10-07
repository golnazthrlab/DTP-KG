"""Combine saved transductive AUROC, F1 and Fusion–MeSH gains in one row."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtpkg.evaluation_stats import stars
from dtpkg.transductive.reporting import (
    COLORS, FIVE_MODELS, ORDER, SHORT, TRANSDUCTIVE_METRICS,
    TRANSDUCTIVE_FAMILY_SIZE, default_paths, input_paths, load_figure_tables,
    separate_output, sha256,
)

STEM = "transductive_fusion_three_panel"
METRICS = TRANSDUCTIVE_METRICS
MODEL_LABELS = {"baseline": "MeSH", "topo_only": "Topology", "fusion": "Fusion",
                "common_neighbors": "DDI neighbours", "degree_product": "DDI degree product"}
TABLE_LABELS = {"baseline": "MeSH", "topo_only": "Topo.",
                "common_neighbors": "DDI CN", "degree_product": "DDI DP"}
MARKERS = ("o", "s", "^", "D", "v")


def gain_text(row, *, compact=False):
    value = f"{row.estimate:+.3f}"
    if compact:
        value = value.replace("+0.", ".").replace("-0.", "−.")
    return value


def significance_mark(p_raw, p_holm):
    if not (np.isfinite(p_raw) and np.isfinite(p_holm)):
        return "NA"
    return stars(p_holm) or ("†" if p_raw < .05 else "")


def draw_lines(ax, tax, summary, comparisons, metric, panel_label):
    x = np.arange(len(ORDER))
    for index, model in enumerate(FIVE_MODELS):
        s = summary[(summary.model == model) & (summary.metric == metric)].set_index("category").loc[ORDER]
        positions = x + .045 * (index - 2)
        means, low, high = (s[column].to_numpy(float) for column in ("estimate", "ci_low", "ci_high"))
        ax.plot(positions, means, marker=MARKERS[index], markersize=4.6,
                color=COLORS[model], linewidth=1.8 if model == "fusion" else 1.25,
                linestyle="-" if model == "fusion" else "--", label=MODEL_LABELS[model])
        # Draw saved interval endpoints even if an approximate interval exceeds [0, 1].
        ax.errorbar(positions, (low + high) / 2, yerr=(high - low) / 2,
                    fmt="none", ecolor=COLORS[model], capsize=2, alpha=.65, elinewidth=.9)
    ax.set_xticks(x, [SHORT[c] for c in ORDER])
    ax.set_xlim(-.4, len(ORDER) - .6)
    ax.set_title(panel_label, loc="left", pad=10, fontsize=13)
    ax.set_ylabel("Score", labelpad=3)
    ax.grid(axis="y", color="#d9dfe4", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

    cells = []
    significant_cells = set()
    for table_row, (model, name) in enumerate(TABLE_LABELS.items(), start=1):
        values = comparisons[(comparisons.metric == metric) & (comparisons.model_b == model)]
        values = values.set_index("category").loc[ORDER]
        row = [name]
        for table_col, value in enumerate(values.itertuples(), start=1):
            mark = significance_mark(value.p_raw, value.p_holm)
            row.append(gain_text(value, compact=True) + (f"\n{mark}" if mark else ""))
            if np.isfinite(value.p_holm) and value.p_holm < .05:
                significant_cells.add((table_row, table_col))
        cells.append(row)
    tax.set_axis_off()
    table = tax.table(cellText=cells, colLabels=["Δ vs", *[SHORT[c] for c in ORDER]],
                      colWidths=[.235, *[.765 / 6] * 6], cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dce2")
        cell.set_linewidth(.5)
        cell.set_facecolor("#edf2f6" if r == 0 else "#f8fafb" if r % 2 else "white")
        cell.PAD = .025
        cell.set_height(.11 if r == 0 else .2225)
        cell.get_text().set_linespacing(.88)
        if r == 0 or (r, c) in significant_cells:
            cell.get_text().set_fontweight("bold")
        if c == 0:
            cell.get_text().set_ha("left")
    return table


def render(source_dir=None, out_dir=None):
    """Return the paper PNG, writing all generated artifacts outside the inputs.

    By default use the public aggregate tables and ``figures/reproduced``.
    Explicit private run roots or figure-data directories are also supported.
    """
    source, out = default_paths(source_dir, out_dir)
    separate_output(source, out)
    inputs = {path: sha256(path) for path in input_paths(source)}
    plotted, comparisons = load_figure_tables(source)
    n_evaluations = int(comparisons.n_folds.iloc[0])
    panel_c = comparisons[comparisons.model_b.eq("baseline") & comparisons.metric.isin(METRICS)].copy()
    matrix = panel_c.pivot(index="metric", columns="category", values="estimate").loc[list(METRICS), ORDER].to_numpy()
    pvalues = panel_c.pivot(index="metric", columns="category", values="p_holm").loc[list(METRICS), ORDER].to_numpy()
    raw_pvalues = panel_c.pivot(index="metric", columns="category", values="p_raw").loc[list(METRICS), ORDER].to_numpy()
    limit = float(np.abs(matrix).max())

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(12.8, 5.6))
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.17],
                               left=.048, right=.965, bottom=.105, top=.825, wspace=.32)
        line_axes, tables = [], []
        for column, (metric, title) in enumerate((("auc", "(a) AUROC"), ("f1", "(b) F1"))):
            inner = grid[column].subgridspec(2, 1, height_ratios=[2.8, 1.80], hspace=.25)
            ax, tax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            tables.append(draw_lines(ax, tax, plotted, comparisons, metric, title))
            line_axes.append(ax)
        heat_grid = grid[2].subgridspec(1, 2, width_ratios=[1, .045], wspace=.085)
        ax, color_ax = fig.add_subplot(heat_grid[0]), fig.add_subplot(heat_grid[1])
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
        for (r, c), value in np.ndenumerate(matrix):
            mark = significance_mark(raw_pvalues[r, c], pvalues[r, c])
            symbol = mark or "·"
            ax.text(c, r, f"{value:+.3f}\n{symbol}", ha="center", va="center", fontsize=9.6,
                    color="white" if abs(value) > .65 * limit else "#111111")
        ax.set_title("(c) Fusion − MeSH", loc="left", pad=10, fontsize=13)
        ax.set_xticks(range(6), [SHORT[c] for c in ORDER])
        ax.set_yticks(range(4), ["AUROC", "F1", "Precision", "Recall"])
        ax.tick_params(labelsize=10, length=0, pad=5)
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(im, cax=color_ax, ticks=[-.10, -.05, 0, .05, .10])
        colorbar.ax.tick_params(labelsize=9)
        colorbar.set_label("Mean paired difference", fontsize=10, labelpad=5)
        fig.legend(*line_axes[0].get_legend_handles_labels(), ncol=5, loc="upper center",
                   bbox_to_anchor=(.50, .995), frameon=False, fontsize=11, columnspacing=1.4,
                   handletextpad=.5, handlelength=1.9)
        fig.text(.50, .920,
                 "† unadjusted p < 0.05, Holm-adjusted p ≥ 0.05    "
                 "* Holm-adjusted p < 0.05    ** Holm-adjusted p < 0.01",
                 ha="center", va="center", fontsize=9.5)
        fig.text(.49, .022, "MeSH-depth pair category", ha="center", fontsize=11)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for table in tables:
            for cell in table.get_celld().values():
                text_bounds = cell.get_text().get_window_extent(renderer)
                cell_bounds = cell.get_window_extent(renderer)
                if text_bounds.width > cell_bounds.width or text_bounds.height > cell_bounds.height:
                    raise ValueError(f"Comparison table text does not fit: {cell.get_text().get_text()}")
        out.mkdir(parents=True, exist_ok=True)
        outputs = []
        for extension in ("png", "pdf"):
            path = out / f"{STEM}.{extension}"
            fig.savefig(path, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=.07)
            outputs.append(path)
        plt.close(fig)

    plotted.to_csv(out / f"{STEM}_summary_ci.csv", index=False)
    comparisons.to_csv(out / f"{STEM}_comparisons_holm.csv", index=False)
    for path, expected_hash in inputs.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"Source changed while drawing: {path}")
    caption = (
        "Transductive performance across six MeSH-depth pair categories. (a) AUROC and (b) F1 for "
        "five models; (c) Fusion–MeSH mean paired differences across AUROC, F1, precision and recall. "
        f"Estimates average {n_evaluations} matched outer evaluations. Bars in "
        "(a,b) are unchanged pointwise 95% approximate corrected-CV intervals. Tables below (a,b) "
        "report Fusion-minus-comparator gains; leading zeros are omitted and significance symbols appear below gains "
        "to fit the compact layout. Holm-significant table gains and their asterisks are bold. "
        "MeSH denotes MeSH-only, Topo. topology-only, DDI CN common DDI "
        "neighbours, and DDI DP DDI degree product. Small horizontal offsets separate models. "
        "Neural thresholded metrics use 0.5; graph-score thresholds were selected on inner validation. "
        f"Panel (c) uses a symmetric zero-centred colour scale of ±{limit:.6f}, based only on its "
        "24 displayed gains. Positive values favour Fusion. Colour encodes effect size; stars "
        "indicate Holm-adjusted significance across all 96 hypotheses (four Fusion "
        "contrasts × six categories × four metrics: AUROC, F1, precision and recall): * p < 0.05, "
        "** p < 0.01. In all panels, † denotes unadjusted p < 0.05 but Holm-adjusted p ≥ 0.05. "
        "Signed values and the colour scale indicate the direction of the differences. "
        "Dots and unmarked table entries meet neither significance threshold. "
        "NA indicates unavailable inference. Intervals are pointwise; inference does not fully "
        "account for shared-drug/network dependence. Non-significance does not establish equivalence. "
        "The reporting family was revised after results were available to include the four reported metrics; "
        "the previous 120-test analysis is archived. The comparison remains post-hoc exploratory. "
        "The revision recalculates only Holm adjustment; models, predictions, raw p-values and "
        "pointwise intervals are unchanged.\n"
    )
    caption_path = out / f"{STEM}_caption.txt"
    caption_path.write_text(caption)
    artifacts = [*outputs, caption_path, out / f"{STEM}_summary_ci.csv", out / f"{STEM}_comparisons_holm.csv"]
    from dtpkg import evaluation_stats
    from dtpkg.transductive import reporting
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        implementation="packaged publication source",
        panels=["AUROC (five models)", "F1 (five models)", "Fusion–MeSH (four metrics)"],
        holm_family_size=TRANSDUCTIVE_FAMILY_SIZE,
        n_outer_evaluations=n_evaluations,
        training_performed=False, predictions_recomputed=False,
        t_tests_checked_from_fold_scores=any(path.name == "fold_metrics.csv" for path in inputs),
        reporting_family_revised_after_results=True,
        heatmap_scale=dict(minimum=-limit, maximum=limit, centre=0),
        inputs={path.name: value for path, value in inputs.items()},
        packaged_code={Path(path).name: sha256(path) for path in
                       (__file__, reporting.__file__, evaluation_stats.__file__)},
        outputs={path.name: sha256(path) for path in artifacts})
    (out / f"{STEM}_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out / f"{STEM}.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    print(render(args.source_dir, args.figures_dir))
