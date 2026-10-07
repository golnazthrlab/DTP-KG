"""Redraw the adjusted-overlap panel from public aggregate statistics."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .reporting import load_tables


def plot_adjusted_overlap(table_dir, output_dir):
    """Verify aggregate inputs and write the manuscript panel, PDF and caption."""
    report = load_tables(table_dir)
    out = Path(output_dir).expanduser().resolve()
    summaries = report["adjusted_summary"].to_dict("records")
    splits = report["adjusted_splits"].to_dict("records")

    def matches(r, scenario, feature):
        return (r['scenario'] == scenario and r['feature'] == feature
                and r['subset'] == 'all_pairs' and r['label_group'] == 'overall'
                and r['statistic'] == 'partial_spearman'
                and r['control_encoding'] == 'unordered_minmax')

    specs = [
        ('transductive', 'shared_protein_count', 'Transductive\nDDI + drug–protein + PPI', '#C45A13'),
        ('transductive', 'biological_shared_protein_count', 'Transductive\nDrug–protein + PPI only', '#246B93'),
        ('seen_unseen', 'biological_shared_protein_count', 'Seen–unseen, DDI-free\nDrug–protein + PPI', '#246B93'),
        ('unseen_unseen', 'biological_shared_protein_count', 'Unseen–unseen, DDI-free\nDrug–protein + PPI', '#246B93'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.1, 5.45))
    fig.subplots_adjust(left=.34, right=.965, top=.79, bottom=.30)
    for i, (scenario, feature, label, color) in enumerate(specs):
        selected = sorted([r for r in splits if matches(r, scenario, feature)],
                          key=lambda r: int(r['split_repeat']))
        summary = [r for r in summaries if matches(r, scenario, feature)]
        if (len(selected) < 2 or len(summary) != 1
                or {int(r['split_repeat']) for r in selected} != set(range(len(selected)))
                or any(r['status'] != 'ok' for r in selected + summary)):
            raise ValueError(f'incomplete or invalid saved estimates: {scenario}/{feature}')
        values = np.array([float(r['mean_rho']) for r in selected])
        mean = float(summary[0]['mean_rho'])
        if (not np.isfinite(values).all() or np.any(np.abs(values) > 1)
                or not math.isclose(float(values.mean()), mean, abs_tol=1e-12)
                or not math.isclose(float(values.std(ddof=1)), float(summary[0]['sd_rho']), abs_tol=1e-12)):
            raise ValueError(f'saved split values and summary disagree: {scenario}/{feature}')
        y = 3-i
        ax.scatter(values, y + np.linspace(.05, .23, len(selected)), s=22,
                   color=color, alpha=.72, linewidth=0, zorder=3)
        ax.scatter([mean], [y-.12], marker='D', s=61, color=color,
                   edgecolor='white', linewidth=.8, zorder=4)
        ax.annotate(f'{mean:.3f}', (mean, y-.12), xytext=(8,-3),
                    textcoords='offset points', color=color, fontsize=10, fontweight='bold')
    ax.set_yticks([3,2,1,0], [s[2] for s in specs])
    ax.set_ylim(-.48, 3.5)
    plotted = np.array([float(row['mean_rho']) for row in splits + summaries])
    if plotted.min() >= -.025 and plotted.max() <= .72:
        ax.set_xlim(-.025, .72)
        ax.set_xticks([0,.1,.2,.3,.4,.5,.6,.7])
    else:
        low, high = min(-.025, plotted.min()-.045), max(.72, plotted.max()+.045)
        ax.set_xticks(np.arange(np.ceil(low*10), np.floor(high*10)+1) / 10)
        ax.set_xlim(low, high)
    ax.set_xlabel('Size-adjusted partial Spearman correlation', labelpad=12)
    ax.axvline(0, color='#888888', lw=.9, zorder=0)
    ax.grid(axis='x', color='#E6E6E6', lw=.7, zorder=0)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#BBBBBB')
    ax.tick_params(axis='y', length=0, pad=12, labelsize=10)
    ax.tick_params(axis='x', length=3, color='#AAAAAA', labelsize=9)
    fig.text(.035,.945,'Shared protein context and prediction scores', fontsize=14, fontweight='bold')
    fig.text(.035,.89,'Exploratory size adjustment · all held-out pairs · topology-only models',
             fontsize=10, color='#444444')
    legend = [Line2D([], [], marker='o', color='#666666', linestyle='', markersize=4,
                     label='Repeat / holdout mean'),
              Line2D([], [], marker='D', color='#333333', linestyle='', markersize=6,
                     label='Overall mean')]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.33,.115),
               ncol=2, frameon=False, fontsize=9, handletextpad=.5, columnspacing=1.4)
    fig.text(.035,.025,
             'Inductive results use the saved DDI-free condition.\n'
             'Dots show descriptive variation, not confidence intervals.',
             fontsize=9.5, color='black', va='bottom', linespacing=1.3)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(out / f'adjusted_overlap_correlations.{ext}', dpi=220,
                    facecolor='white', bbox_inches='tight')
    plt.close(fig)
    caption = '''Exploratory size-adjusted associations between shared protein-neighborhood overlap and topology-only prediction scores. Neighborhoods contain proteins within at most two edges of each drug. Transductive full-graph overlap permits DDI-mediated paths; the second transductive row restricts the overlap measurement to drug–protein/PPI paths while retaining the same predictions. Inductive results use the saved DDI-free condition, where both overlap definitions coincide. Within each fit, overlap counts and scores were ranked and separately regressed on an intercept and the ranked smaller and larger endpoint protein-neighborhood sizes; the residuals were correlated without reranking. All held-out pairs were included. Small dots represent five transductive repetition means after averaging folds, or five inductive holdout means after averaging training seeds. Diamonds indicate equally weighted means of those five values. Shared drugs and overlapping splits create dependence; dots show descriptive variation, not confidence intervals. The cohorts and fitted models differ across evaluation settings. Adjustments were exploratory; unadjusted correlations and additional sensitivity analyses should be reported in the supplement. Data source: saved case-study analysis dated 21 September 2026, distinct from the later inductive experiment retaining known DDI context.\n'''
    counts = report["protocol"]["split_repeats"]
    if len(counts['transductive']) != 5 or len(counts['inductive']) != 5:
        caption = caption.replace("five transductive repetition means", f"{len(counts['transductive'])} transductive repetition means")
        caption = caption.replace("five inductive holdout means", f"{len(counts['inductive'])} inductive holdout means")
        caption = caption.replace("equally weighted means of those five values", "equally weighted means of those split values")
    if report["protocol"]["snapshot_date"] != "2026-09-21":
        caption = caption.replace("Data source: saved case-study analysis dated 21 September 2026, distinct from the later inductive experiment retaining known DDI context.", "Data source: locally regenerated case-study analysis with DDI-free inductive graphs.")
    (out / 'adjusted_overlap_correlations_caption.txt').write_text(caption)
    return {ext: out / f"adjusted_overlap_correlations.{ext}" for ext in ("png", "pdf")} | {
        "caption": out / "adjusted_overlap_correlations_caption.txt"}
