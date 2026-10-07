"""Render the selected seen–unseen AUROC and F1 figures from aggregate tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from dtpkg.inductive.reporting import (
    DEPTH, MODELS, SPLIT_IDS, default_paths, load_report, separate_output, sha256,
)

METRICS = {"auc": "AUROC", "f1": "F1"}
MODEL_LABELS = {"baseline": "MeSH", "topo_only": "Topology", "fusion": "Fusion"}
COLORS = {"baseline": "#737B86", "topo_only": "#D55E00", "fusion": "#0072B2"}
MARKERS = {"baseline": "o", "topo_only": "s", "fusion": "^"}
DEPTH_LABELS = ("Low–Low", "Low–Mid", "Low–Deep", "Mid–Mid", "Mid–Deep", "Deep–Deep")
SIGNIFICANCE_LEGEND = (r'† $p_{\mathrm{raw}}<0.05\leq p_{\mathrm{Holm}}$'
                       r'    * $p_{\mathrm{Holm}}<0.05$'
                       '    Bold: either criterion')

def _ptext(value, precision=3):
    return 'NA' if not np.isfinite(value) else '<0.001' if value < .001 else f'{value:.{precision}f}'

def _pstars(value):
    """Mark significance using the unrounded displayed test's p-value."""
    if not np.isfinite(value):
        return ''
    return '***' if value < .001 else '**' if value < .01 else '*' if value < .05 else ''

def _draw_point(ax, x, row, model, spread=.035):
    values = row[list(SPLIT_IDS)].to_numpy(float)
    ax.scatter(x + np.linspace(-spread, spread, 3), values, facecolors='none', edgecolors=COLORS[model],
               s=20, linewidths=.8, alpha=.65, zorder=3)
    if np.isfinite(row.estimate):
        ax.errorbar(x, row.estimate, yerr=row.sd_across_holdouts, color=COLORS[model],
            marker=MARKERS[model], markersize=6, capsize=3, linewidth=0, elinewidth=1.3, zorder=4)
    else:
        ax.text(x, .02, 'NA', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)

def _limits(data):
    bounds = np.r_[data.estimate - data.sd_across_holdouts, data.estimate + data.sd_across_holdouts,
                   data[list(SPLIT_IDS)].to_numpy().ravel()]
    bounds = bounds[np.isfinite(bounds)]
    if not len(bounds):
        return (0., 1.)
    span = max(float(np.ptp(bounds)), .18)
    return float(bounds.min() - .13 * span), float(bounds.max() + .25 * span)

def one(frame, **conditions):
    selected = frame
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {conditions}; found {len(selected)}")
    return selected.iloc[0]

def style_axis(ax, metric, limits):
    ax.set_ylim(*limits)
    ax.set_ylabel(METRICS[metric])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#e1e6eb", linewidth=.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

def significance_mark(p_raw, p_holm):
    """Use unrounded p-values, with stars taking precedence over the dagger."""
    if not (np.isfinite(p_raw) and np.isfinite(p_holm)):
        return ''
    return _pstars(p_holm) or ('†' if p_raw < .05 else '')

def _table(ax, rows, labels, widths, significant=()):
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=labels, colWidths=widths,
                     cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(.20 if row == 0 else .40)
        cell.set_edgecolor('#d6dce2')
        cell.set_linewidth(.6)
        cell.set_facecolor('#edf2f6' if row == 0 else '#f8fafb' if row % 2 else 'white')
        cell.PAD = .035
        if row == 0 or (row, column) in significant:
            cell.get_text().set_fontweight('bold')
        if column == 0:
            cell.get_text().set_ha('left')
    return table

def _overall_table(contrasts):
    rows, significant = [], set()
    for row_index, comparator in enumerate(('baseline', 'topo_only'), start=1):
        row = one(contrasts, model_b=comparator)
        mark = significance_mark(row.p_raw, row.p_holm)
        rows.append([MODEL_LABELS[comparator],
                     f'{row.estimate:+.3f} ± {row.sd_across_holdouts:.3f}',
                     _ptext(row.p_holm) + mark])
        if row.p_raw < .05 or row.p_holm < .05:
            significant.update(((row_index, 1), (row_index, 2)))
    return rows, ['Fusion vs', 'Gain ± SD', r'$p_{\mathrm{Holm}}$'], [.29, .46, .25], significant

def _depth_table(contrasts):
    rows, significant = [], set()
    for row_index, comparator in enumerate(('baseline', 'topo_only'), start=1):
        cells = [MODEL_LABELS[comparator]]
        for column, category in enumerate(DEPTH, start=1):
            row = one(contrasts, category=category, model_b=comparator)
            mark = significance_mark(row.p_raw, row.p_holm)
            if not (np.isfinite(row.p_raw) and np.isfinite(row.p_holm)):
                mark = 'NA'
            cells.append(f'{row.estimate:+.3f}±{row.sd_across_holdouts:.3f}'
                         + (f'\n{mark}' if mark else ''))
            if row.p_raw < .05 or row.p_holm < .05:
                significant.add((row_index, column))
        rows.append(cells)
    return rows, ['Fusion vs', *DEPTH_LABELS], [.12, *[.88 / 6] * 6], significant

def figure_caption(metric='auc'):
    introduction = (
        f'Seen–unseen inductive {METRICS[metric]} performance. '
        'Panel (a) shows overall performance; panel (b) shows performance by MeSH-depth pair category. '
    )
    return (
        introduction +
        'Every pair contains one held-out drug and one seen partner, and all DDI edges are absent from the feature graphs. '
        'Filled markers show the mean across three drug holdouts after averaging three training seeds within each holdout; '
        'open circles show individual holdout means. Error bars are sample SD across holdouts, not confidence intervals. '
        'Tables show paired fusion-minus-comparator mean gains and their sample SD. '
        'Overall p-values are Holm-corrected across four primary tests: AUROC and F1, each comparing fusion with MeSH and topology. '
        f'The unplotted {"F1" if metric == "auc" else "AUROC"} comparisons remain in the overall family. '
        'Depth tables show gain ± SD with significance symbols; numerical p-values are retained in the '
        'accompanying comparison exports. Depth adjustment uses the existing separate family of 60 tests: five metrics '
        '(AUROC, F1, precision, recall and accuracy), six depth categories and two fusion comparators. '
        'The unplotted metrics remain in this family. An asterisk (*) identifies Holm-adjusted p < 0.05. '
        'In all panels, † denotes unadjusted p < 0.05 but '
        'Holm-adjusted p ≥ 0.05. Bold numerical entries identify comparisons significant before and/or '
        'after Holm adjustment, including dagger-marked comparisons; bold emphasis alone does not imply '
        'Holm significance. Unmarked comparisons meet neither '
        'significance threshold; NA denotes unavailable inference. Symbols and bold emphasis use unrounded p-values. '
        'The underlying tests are unchanged approximate overlap-corrected, two-sided paired t-tests '
        'on three seed-averaged holdout differences (df=2). This display revision was made after inspecting '
        'results and retains the existing hypothesis families. Depth comparisons remain exploratory. '
        + ('F1 uses the fixed threshold 0.5. ' if metric == 'f1' else '')
    ).strip() + '\n'

def _draw_panels(tables, metric, fig, grid, titles):
    """Keep plot bounds, table bounds and model encodings aligned in both panels."""
    summary = tables['summary'].query('metric == @metric')
    overall = tables['overall'].query('metric == @metric')
    depth = tables['depth'].query('metric == @metric')
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    table_axes = [fig.add_subplot(grid[1, i]) for i in range(2)]
    limits = _limits(summary)
    for ax in axes:
        style_axis(ax, metric, limits)
        ax.tick_params(axis='both', labelsize=10.5)
    axes[1].set_ylabel('')

    overall_data = summary[summary.group_kind.eq('overall')]
    for index, model in enumerate(MODELS):
        _draw_point(axes[0], index, one(overall_data, model=model), model)
    axes[0].set_xticks(range(3), [MODEL_LABELS[m] for m in MODELS])
    axes[0].set_xlim(-.45, 2.45)
    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].set_title(titles[0], loc='left', fontsize=13, fontweight='bold', pad=13)

    depth_data = summary[summary.group_kind.eq('depth')]
    for index, category in enumerate(DEPTH):
        for offset, model in enumerate(MODELS):
            _draw_point(axes[1], index + .20 * (offset - 1),
                                one(depth_data, category=category, model=model), model, .025)
    axes[1].set_xticks(range(6), DEPTH_LABELS)
    axes[1].set_xlim(-.5, 5.5)
    axes[1].set_xlabel('MeSH-depth pair category', fontsize=11)
    axes[1].set_title(titles[1], loc='left', fontsize=13, fontweight='bold', pad=13)

    rows, labels, widths, overall_significant = _overall_table(overall)
    overall_table = _table(table_axes[0], rows, labels, widths, overall_significant)
    rows, labels, widths, significant = _depth_table(depth)
    depth_table = _table(table_axes[1], rows, labels, widths, significant)
    table_axes[0].set_title('Overall comparisons', loc='left', fontsize=10.5, pad=9)
    table_axes[1].set_title('Depth comparisons · gain ± SD', loc='left', fontsize=10.5, pad=9)
    return axes, table_axes, (overall_table, depth_table), significant

def build_figure(tables, metric):
    fig = plt.figure(figsize=(14.4, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[3.2, 1.5],
                           left=.055, right=.99, top=.79, bottom=.12,
                           wspace=.16, hspace=.50)
    panels = _draw_panels(tables, metric, fig, grid, ('(a) Overall', '(b) By MeSH depth'))

    fig.suptitle(f'Seen–unseen · {METRICS[metric]}', fontsize=17, fontweight='bold', y=.975)
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle='none',
                      markersize=7, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .929),
               ncol=3, frameon=False, fontsize=12)
    fig.text(.99, .052, SIGNIFICANCE_LEGEND,
             ha='right', va='center', fontsize=10.5)
    return fig, *panels

def _check_layout(fig, axes, table_axes, tables):
    """Check actual rendered text fits its cells and panel geometry is aligned."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for left, right in (axes, table_axes):
        a, b = left.get_position(), right.get_position()
        if not np.allclose([a.y0, a.y1, b.width / a.width], [b.y0, b.y1, 2]):
            raise ValueError('Panels must have aligned bounds and 1:2 widths')
    for table in tables:
        for key, cell in table.get_celld().items():
            outer, inner = cell.get_window_extent(renderer), cell.get_text().get_window_extent(renderer)
            if inner.x0 < outer.x0 or inner.x1 > outer.x1 or inner.y0 < outer.y0 or inner.y1 > outer.y1:
                raise ValueError(f'Table text overflows cell {key}: {cell.get_text().get_text()}')


def render_figures(source=None, out_dir=None):
    """Validate scores and regenerate both figures; return paths keyed by metric."""
    source, output = default_paths(source, out_dir)
    separate_output(source, output)
    tables = load_report(source)
    inputs = {path: sha256(path) for path in source.iterdir() if path.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": "#aaaaaa", "pdf.fonttype": 42, "ps.fonttype": 42}):
        for metric in METRICS:
            fig, axes, table_axes, figure_tables, _ = build_figure(tables, metric)
            try:
                _check_layout(fig, axes, table_axes, figure_tables)
                stem = f'seen_unseen_{"auroc" if metric == "auc" else "f1"}'
                path = output / (stem + ".png")
                fig.savefig(path, dpi=300, facecolor="white")
                (output / (stem + "_caption.txt")).write_text(figure_caption(metric))
                paths[metric] = path
            finally:
                plt.close(fig)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Plotting input changed during rendering: {path.name}")
    (output / "reproduction.json").write_text(json.dumps(dict(
        inputs={path.name: digest for path, digest in inputs.items()},
        outputs={path.name: sha256(path) for path in paths.values()},
        metrics=list(METRICS), overall_family_size=4, depth_family_size=60,
        training_performed=False, predictions_recomputed=False,
        aggregate_tests_recomputed=True), indent=2, sort_keys=True) + "\n")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Aggregate tables from write_report (defaults to public tables)")
    parser.add_argument("--out-dir", type=Path, help="Output directory (defaults to figures/reproduced)")
    args = parser.parse_args(argv)
    for metric, path in render_figures(args.source, args.out_dir).items():
        print(f"{METRICS[metric]}: {path}")


if __name__ == "__main__":
    main()
