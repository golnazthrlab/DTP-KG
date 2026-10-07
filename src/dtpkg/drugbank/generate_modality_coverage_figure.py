"""Generate the study modality-coverage figure from locally prepared field flags.

The population is all DrugBank small-molecule and biotech entries. Fields are
ranked by biotech coverage, with structural baselines shown separately.
"""
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtpkg.project_paths import DATA_DIR, EXPERIMENTS_DIR

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def generate_modality_coverage(coverage, output_dir, *, pdf=False):
    """Render the published layout from a coverage DataFrame or CSV path.

    Only the selected PNG is written by default. Return its path.
    """
    # Isolate each figure from notebook themes and earlier plotting calls.
    with plt.rc_context(rc=plt.rcParamsDefault):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        OUT_PNG = output_dir / "modality_coverage_table.png"
        OUT_PDF = output_dir / "modality_coverage_table.pdf"
        df = pd.read_csv(coverage) if isinstance(coverage, (str, Path)) else coverage.copy()
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].map({'True': True, 'False': False}).fillna(df[c])

        n_sm  = int((df['type'] == 'small molecule').sum())
        n_bio = int((df['type'] == 'biotech').sum())

        KG_EDGE_FIELDS = ['targets', 'drug-interactions', 'enzymes', 'transporters',
                          'carriers', 'pathways', 'reactions']
        df['any_KG'] = df[KG_EDGE_FIELDS].astype(bool).any(axis=1)

        # (cache column, display label, row kind)
        FIELDS = [
            ('smiles',              'SMILES',                    'other'),
            ('any_KG',              'Any KG-representable edge', 'hero_kg'),
            ('targets',             'Targets',                   'kg_member'),
            ('mesh-categories',     'MeSH categories',           'hero_mesh'),
            ('pdb-entries',         'PDB entries',               'other'),
            ('drug-interactions',   'Drug–drug interactions',    'kg_member'),
            ('mechanism-of-action', 'Mechanism of action',       'other'),
            ('atc-codes',           'ATC codes',                 'other'),
            ('enzymes',             'Enzymes',                   'kg_member'),
            ('transporters',        'Transporters',              'kg_member'),
            ('reactions',           'Reactions',                 'kg_member'),
            ('carriers',            'Carriers',                  'kg_member'),
            ('pathways',            'Pathways',                  'kg_member'),
            ('aa-sequence',         'Amino-acid sequence',       'other'),
        ]
        rows = []
        for col, label, kind in FIELDS:
            s = df[col].astype(bool)
            rows.append({
                'label': label,
                'kind' : kind,
                'sm'   : 100 * s[df['type']=='small molecule'].mean(),
                'bio'  : 100 * s[df['type']=='biotech'].mean(),
            })
        cov = pd.DataFrame(rows)

        # Structural baselines: SMILES and AA sequence are by construction modality-
        # specific (each is 0 % on one side). They are shown together below the divider
        # as paired baselines, not within the main ranking. Treating them symmetrically
        # avoids the inconsistency of including AA sequence in the ranking while
        # excluding it as "asymmetric".
        APPENDIX_LABELS = ['SMILES', 'Amino-acid sequence']

        # Main ranking: everything except the structural baselines, ranked by Bio
        # coverage descending (SM coverage as tiebreaker).
        main_pool = cov[~cov['label'].isin(APPENDIX_LABELS)].copy()
        main_pool = (main_pool.sort_values(['bio', 'sm'], ascending=[False, False])
                               .reset_index(drop=True))
        main_pool['rank']        = np.arange(1, len(main_pool) + 1)
        main_pool['is_appendix'] = False
        cov_top = main_pool.head(10).copy()

        # Append both structural baselines (Bio-desc order: AA sequence first, then SMILES)
        appendix_rows = (cov[cov['label'].isin(APPENDIX_LABELS)]
                           .sort_values('bio', ascending=False)
                           .reset_index(drop=True))
        appendix_rows['rank']        = np.nan
        appendix_rows['is_appendix'] = True
        cov_top = pd.concat([cov_top, appendix_rows], ignore_index=True)

        # ── Colours ───────────────────────────────────────────────────────────────
        SM_COLOR  = '#2166ac'
        BIO_COLOR = '#d6604d'
        GRID      = '#e6e6e6'
        HERO_MESH = '#fdf3d6'   # cream — MeSH (semantic backbone)
        HERO_KG   = '#b8d4ea'   # clear pale blue — KG composite (distinct from KG_MEMBER)
        KG_MEMBER = '#ececec'   # neutral grey — KG contributing field (no blue tint)

        plt.rcParams.update({
            'font.family'      : 'sans-serif',
            'axes.spines.top'  : False,
            'axes.spines.right': False,
            'axes.spines.left' : False,
        })

        # ── Figure ────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12.5, 9.4))
        fig.patch.set_facecolor('white')

        n = len(cov_top)
        # y-positions: top 10 at y = 10, 9, …, 1; appendix rows stacked below the gap
        GAP = 1.0
        n_main      = 10
        n_appendix  = n - n_main
        yy_main     = np.arange(1, n_main + 1)[::-1]               # 10, 9, …, 1
        # Appendix rows at y = -1, -2, …  (each 1 unit below the previous)
        yy_appendix = -GAP - np.arange(n_appendix)
        yy = np.concatenate([yy_main, yy_appendix])

        # Row backgrounds
        KIND_BG = {'hero_mesh': HERO_MESH, 'hero_kg': HERO_KG,
                   'kg_member': KG_MEMBER, 'other': None}
        for i, row in cov_top.iterrows():
            bg = KIND_BG[row['kind']]
            if bg is not None:
                ax.axhspan(yy[i] - 0.45, yy[i] + 0.45, color=bg, zorder=0)

        # Grid
        ax.xaxis.grid(True, color=GRID, lw=0.9, zorder=1)
        ax.set_axisbelow(True)

        # Truncation indicator + visible separator between the ranked block and the
        # appendix row. Vertical ellipsis = "rows omitted here"; the solid line below
        # the explanatory text is the divider the caption references.
        DOTS_Y = 0.18
        TEXT_Y = -0.18
        SEP_Y  = -0.55   # solid divider line directly above the SMILES row band

        # Visible separator line — runs the full chart width
        ax.axhline(SEP_Y, color='#666666', lw=1.1, zorder=1.5)

        # Vertical ellipsis on the y-axis (label column)
        ax.text(-3, DOTS_Y, '⋮',
                fontsize=26, color='#7a7a7a', weight='bold',
                va='center', ha='right')

        # Mirror the ellipsis at a few positions across the chart to suggest
        # the data continues but is being skipped
        for x_pos in [12, 35, 60, 85]:
            ax.text(x_pos, DOTS_Y, '⋮',
                    fontsize=18, color='#c8c8c8',
                    va='center', ha='center')

        # Brief italic annotation naming what's omitted (lower-Bio-coverage fields)
        ax.text(50, TEXT_Y,
                'Modalities with lower biotech coverage omitted '
                '(pathways, PDB entries)',
                fontsize=9.5, style='italic', color='#777777',
                va='center', ha='center')

        # Connectors
        for i, row in cov_top.iterrows():
            ax.plot([row['bio'], row['sm']], [yy[i], yy[i]],
                    color='#bbbbbb', lw=2.2, solid_capstyle='round', zorder=2)

        # Dots — larger for the top-10 layout
        ax.scatter(cov_top['sm'],  yy, color=SM_COLOR,  s=110, zorder=4,
                   edgecolors='white', linewidths=1.2)
        ax.scatter(cov_top['bio'], yy, color=BIO_COLOR, s=110, zorder=4,
                   edgecolors='white', linewidths=1.2)

        # Value labels — bigger
        for i, row in cov_top.iterrows():
            y    = yy[i]
            sm_v = row['sm']
            bi_v = row['bio']
            leader, trailer = (sm_v, bi_v) if sm_v >= bi_v else (bi_v, sm_v)
            leader_col, trailer_col = (SM_COLOR, BIO_COLOR) if sm_v >= bi_v else (BIO_COLOR, SM_COLOR)
            ax.text(leader + 1.8, y, f'{leader:.0f}',
                    va='center', ha='left', fontsize=12, color=leader_col, fontweight='bold')
            if trailer >= 3.5:
                ax.text(trailer - 1.8, y, f'{trailer:.0f}',
                        va='center', ha='right', fontsize=12, color=trailer_col)
            else:
                ax.text(leader + 9, y, f'· {trailer:.1f}',
                        va='center', ha='left', fontsize=10.5, color=trailer_col)

        # Y-axis labels: name only. Bold heroes; tint KG-member labels navy.
        ax.set_yticks(yy)
        ax.set_yticklabels(cov_top['label'], fontsize=12)
        for tick, kind, is_app in zip(ax.get_yticklabels(), cov_top['kind'], cov_top['is_appendix']):
            if is_app:
                tick.set_color('#555555'); tick.set_fontstyle('italic')
            elif kind == 'hero_mesh':
                tick.set_fontweight('bold'); tick.set_color('#7a5a00')
            elif kind == 'hero_kg':
                tick.set_fontweight('bold'); tick.set_color('#1a4670')
            elif kind == 'kg_member':
                tick.set_color('#1a4670')
            else:
                tick.set_color('#222222')


        # Y/X limits — extend Y to include the appendix row + gap
        ax.set_ylim(yy_appendix.min() - 0.7, yy_main.max() + 0.7)
        # X-axis
        ax.set_xlim(-3, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0', '25', '50', '75', '100 %'], fontsize=11, color='#444444')
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0, pad=5)

        # Legend — vertical, to the right of the axes, outside the plotting area
        legend_handles = [
            Line2D([0], [0], marker='o', color=SM_COLOR,  ls='', ms=11, mec='white', mew=1.2,
                   label=f'Small molecule\n(n = {n_sm:,})'),
            Line2D([0], [0], marker='o', color=BIO_COLOR, ls='', ms=11, mec='white', mew=1.2,
                   label=f'Biotech\n(n = {n_bio:,})'),
            Patch(facecolor=HERO_MESH, edgecolor='#cccccc',
                  label='MeSH —\nsemantic backbone'),
            Patch(facecolor=HERO_KG, edgecolor='#cccccc',
                  label='KG — drug-graph\nedges (any)'),
            Patch(facecolor=KG_MEMBER, edgecolor='#cccccc',
                  label='KG contributing\nfield'),
        ]
        ax.legend(handles=legend_handles,
                  loc='center left', bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=11, handletextpad=0.7,
                  labelspacing=1.4, borderaxespad=0)

        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
        if pdf:
            plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
        plt.close()


        return OUT_PNG


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "drugbank" / "coverage_analysis_cache.csv")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "01_data_exploration" / "figures" / "reproduced")
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF.")
    args = parser.parse_args()
    print(generate_modality_coverage(args.input, args.output_dir, pdf=args.pdf))


if __name__ == "__main__":
    main()
