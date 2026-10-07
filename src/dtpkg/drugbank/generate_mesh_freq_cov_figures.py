"""Generate the published MeSH term-frequency and cumulative-coverage panels.

Each curve is an embedding scope, defined by term depth and inclusive frequency
bounds. Frequencies count distinct approved drugs after branch-D ancestor
closure. Faded curves include all scope terms; bold curves use retained terms.
"""
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtpkg.project_paths import DATA_DIR, EXPERIMENTS_DIR

import json
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from dtpkg.mesh_scopes import (SCOPES, coverage_curve, load_annotations, retained_coverage_curve,
                         scope_counts, scope_table, term_frequencies)

def generate_mesh_panels(annotations, output_dir, *, pdf=False, export_data=False, verify_embeddings=False, combined=False):
    """Render both panels from an annotation DataFrame or CSV path.

    Return a mapping of figure stems to PNG paths. Paper snapshot counts are
    validated; checking saved embedding files is optional.
    """
    # Isolate each figure from notebook themes and earlier plotting calls.
    with plt.rc_context(rc=plt.rcParamsDefault):
        FIGS = Path(output_dir)
        FIGS.mkdir(parents=True, exist_ok=True)
        DATA_OUT = FIGS / "data"
        if export_data:
            DATA_OUT.mkdir(parents=True, exist_ok=True)
        paths = {}
        EXPECTED = {"Shallow": (1416, 2694), "Intermediate": (2128, 3125), "Full": (2183, 2994)}

        CUT_GREY = "#777777"
        INK = "#333333"
        DASH = (0, (4, 3))

        # Reconstruct each scope; stored-embedding checks are optional.
        ann = load_annotations(annotations) if isinstance(annotations, (str, Path)) else annotations.copy()
        freq, cov, rcov, counts = {}, {}, {}, {}
        for s in SCOPES:
            freq[s.name] = term_frequencies(ann, s)
            cov[s.name] = coverage_curve(ann, s)
            rcov[s.name] = retained_coverage_curve(ann, s)
            counts[s.name] = scope_counts(ann, s, check_stored=verify_embeddings)
            got = (counts[s.name]["terms_retained"], counts[s.name]["embedding_drugs_with_a_retained_term"])
            assert got == EXPECTED[s.name], f"{s.name}: reconstruction {got} != expected {EXPECTED[s.name]}"
            if verify_embeddings:
                assert counts[s.name]["stored_embedding_matches_reconstruction"], \
                    f"{s.name}: stored embedding is missing or its drug set differs from reconstruction"
            if export_data:
                # export the numbers behind the panels
                f = freq[s.name]
                pd.DataFrame({"tree_number": f.index, "drugs_per_term": f.values,
                              "retained": (f.values >= s.freq_min) & (f.values <= s.freq_max)}) \
                    .sort_values("drugs_per_term", ascending=False) \
                    .to_csv(DATA_OUT / f"mesh_term_frequency_{s.level_key}.csv", index=False)
                rank, frac, kept = cov[s.name]
                sub = scope_table(ann, s)
                population = sub["drugbank_id"].nunique()
                order = sub.groupby("tree_number")["drugbank_id"].nunique().sort_values(ascending=False, kind="stable")
                pd.DataFrame({"rank": rank, "tree_number": order.index, "drugs_per_term": order.values,
                              "cumulative_covered_drugs": np.round(frac * population).astype(int),
                              "coverage_fraction": frac, "retained": kept,
                              "denominator_drugs_with_any_term_in_depth_range": population}) \
                    .to_csv(DATA_OUT / f"mesh_coverage_all_terms_{s.level_key}.csv", index=False)
                rrank, rfrac = rcov[s.name]
                pd.DataFrame({"rank_among_retained": rrank,
                              "cumulative_covered_drugs": np.round(rfrac * population).astype(int),
                              "coverage_fraction": rfrac,
                              "denominator_drugs_with_any_term_in_depth_range": population}) \
                    .to_csv(DATA_OUT / f"mesh_coverage_retained_terms_{s.level_key}.csv", index=False)

        ALL_MAX = max(int(f.max()) for f in freq.values())
        ALL_MAX_RANK = max(len(f) for f in freq.values())

        plt.rcParams.update({
            "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
            "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6.8,
        })


        def _tag(ax, x, y, text, edge, ha, va, fontsize, transform=None):
            return ax.text(x, y, text, transform=transform or ax.transData, ha=ha, va=va,
                           fontsize=fontsize, color=INK, zorder=6,
                           bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=edge, lw=0.9))


        def label_cutoff_lines(ax, fontsize=7.5, y=0.975, min_gap_dec=0.12):
            """Value tags on the dashed bound lines; near-coincident caps go to opposite sides."""
            caps = sorted(SCOPES, key=lambda s: s.freq_max)
            ha = {s.name: "center" for s in caps}
            for prev, cur in zip(caps, caps[1:]):
                if np.log10(cur.freq_max) - np.log10(prev.freq_max) < min_gap_dec:
                    ha[prev.name], ha[cur.name] = "right", "left"
            _tag(ax, SCOPES[0].freq_min, y, f"{SCOPES[0].freq_min}", CUT_GREY, "center", "top",
                 fontsize, transform=ax.get_xaxis_transform())
            for s in caps:
                _tag(ax, s.freq_max, y, f"{s.freq_max}", s.color, ha[s.name], "top", fontsize,
                     transform=ax.get_xaxis_transform())


        def draw_freq(ax, compact=True):
            grid_log = np.linspace(0, np.log10(ALL_MAX) + 0.15, 400)
            grid_x = 10 ** grid_log
            peak = 0.0
            for s in SCOPES:
                lv = np.log10(freq[s.name].values.astype(float))
                d = gaussian_kde(lv, bw_method=0.25)(grid_log)
                peak = max(peak, d.max())
                ax.fill_between(grid_x, d, color=s.color, alpha=0.08, zorder=2)
                ax.plot(grid_x, d, color=s.color, lw=2.0 if compact else 2.6, zorder=3)
            lo = {s.freq_min for s in SCOPES}
            assert len(lo) == 1, "scopes share one lower bound in this figure design"
            ax.axvline(lo.pop(), color=CUT_GREY, lw=1.0, ls=DASH, zorder=4)
            for s in SCOPES:
                ax.axvline(s.freq_max, color=s.color, lw=1.1 if compact else 1.8, ls=DASH, alpha=0.95, zorder=4)
            ax.set_xscale("log")
            ax.set_xlim(0.8, ALL_MAX * 1.1)
            ax.set_ylim(0, peak * 1.12)
            ax.set_xlabel("Drugs per MeSH term (within scope depth range)")
            ax.set_ylabel("Term density")
            ax.grid(True, which="both", color="#eeeeee", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelleft=False, length=0)
            ax.tick_params(axis="x", length=2, pad=2)
            label_cutoff_lines(ax, fontsize=7.5 if compact else 9.5)


        def draw_cov(ax, compact=True):
            for s in SCOPES:
                rank, frac, _ = cov[s.name]
                ax.plot(rank, 100 * frac, color=s.color, lw=1.0 if compact else 1.4, alpha=0.22, zorder=2)
                rrank, rfrac = rcov[s.name]
                ax.plot(rrank, 100 * rfrac, color=s.color, lw=2.0 if compact else 3.0, zorder=3,
                        solid_capstyle="round")
                ax.scatter([rrank[-1]], [100 * rfrac[-1]], color=s.color, s=22 if compact else 40,
                           zorder=4, edgecolors="white", linewidths=0.8)
                _tag(ax, rrank[-1] * 1.12, 100 * rfrac[-1], f"{100*rfrac[-1]:.1f}%", s.color,
                     "left", "center", 7 if compact else 8.5)
            ax.set_xscale("log")
            ax.set_xlim(0.8, ALL_MAX_RANK * 2.6)
            ax.set_xlabel("MeSH terms by decreasing frequency (rank)")
            ax.set_ylabel("Drugs covered (% of 3,342 with a term in range)")
            ax.grid(True, which="both", color="#eeeeee", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(0, 105)
            ax.tick_params(axis="both", length=2, pad=2)


        def scope_key(target, where, anchor, extra_handles=(), title=None, fontsize=7, ncol=1):
            handles = [Line2D([0], [0], color=s.color, lw=3,
                              label=f"{s.name}: {s.depth_label}, {s.freq_min}–{s.freq_max} drugs/term")
                       for s in SCOPES]
            handles += list(extra_handles)
            return target.legend(handles=handles, loc=where, bbox_to_anchor=anchor, ncol=ncol,
                                 frameon=True, framealpha=0.95, edgecolor="#dddddd", borderpad=0.4,
                                 labelspacing=0.35, fontsize=fontsize, title=title,
                                 title_fontsize=fontsize, handlelength=1.3, columnspacing=0.9)


        def lower_bound_handle(lw=1.0):
            return Line2D([0], [0], color=CUT_GREY, lw=lw, ls=DASH, label="shared lower bound (inclusive)")


        def dropped_handle(lw=1.0):
            return Line2D([0], [0], color="#999999", lw=lw, alpha=0.6,
                          label="faded: all terms incl. dropped generic ones")


        def save(fig, stem):
            fig.savefig(FIGS / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
            if pdf:
                fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths[stem] = FIGS / f"{stem}.png"


        # 1. frequency panel
        fig, ax = plt.subplots(figsize=(5.0, 4.0)); fig.patch.set_facecolor("white")
        draw_freq(ax)
        # Right half of the panel is empty (density tail); the bound tags sit at the top.
        scope_key(ax, "upper right", (0.99, 0.86), extra_handles=[lower_bound_handle()],
                  title="Embedding scope (term depth, retained frequency)")
        plt.tight_layout(); save(fig, "mesh_term_freq_panel")

        # 2. coverage panel -- legend below the axes; the curves fill the plotting area.
        fig, ax = plt.subplots(figsize=(5.0, 4.6)); fig.patch.set_facecolor("white")
        draw_cov(ax)
        scope_key(fig, "lower center", (0.5, 0.0), extra_handles=[dropped_handle()],
                  title="Embedding scope · bold = retained vocabulary only", ncol=2)
        plt.tight_layout(rect=[0, 0.17, 1, 1]); save(fig, "mesh_term_cov_panel")

        if combined:
            # 3. combined
            fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4)); fig.patch.set_facecolor("white")
            draw_freq(axes[0], compact=False)
            scope_key(axes[0], "upper right", (0.99, 0.86), extra_handles=[lower_bound_handle(1.5)],
                      title="Embedding scope (term depth, retained frequency)", fontsize=8.6)
            draw_cov(axes[1], compact=False)
            scope_key(axes[1], "lower right", (0.985, 0.03), extra_handles=[dropped_handle(1.4)],
                      title="Embedding scope · bold = retained vocabulary only", fontsize=7.8)
            plt.tight_layout(); save(fig, "mesh_term_combined")

        if export_data:
            # ── summary ──
            summary = {"note": ("curves are embedding scopes (term-depth + frequency filters over the whole "
                                "annotation table), not the Low/Mid/Deep drug knowledge categories "
                                "(deepest level ≤5 / 6–7 / 8–10)"),
                       "scopes": []}
            for s in SCOPES:
                rank, frac, kept = cov[s.name]
                idx = np.flatnonzero(kept)
                c = dict(counts[s.name])
                rrank, rfrac = rcov[s.name]
                c.update(retained_rank_range_in_all_terms=[int(rank[idx[0]]), int(rank[idx[-1]])],
                         coverage_all_terms_at_last_retained=round(float(frac[idx[-1]]), 4),
                         coverage_all_terms=round(float(frac[-1]), 4),
                         coverage_retained_vocabulary=round(float(rfrac[-1]), 4),
                         coverage_retained_vocabulary_drugs=int(round(rfrac[-1] * c["population_drugs_with_any_term_in_depth_range"])))
                summary["scopes"].append(c)
            (DATA_OUT / "mesh_scope_figure_summary.json").write_text(json.dumps(summary, indent=2, default=str))

        return paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "mesh" / "extended_drug_info.csv")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "01_data_exploration" / "figures" / "reproduced")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--export-data", action="store_true", help="Write local aggregate plotting tables.")
    parser.add_argument("--verify-embeddings", action="store_true", help="Also check locally stored embedding drug sets.")
    parser.add_argument("--combined", action="store_true", help="Also generate the unsubmitted combined layout.")
    args = parser.parse_args()
    outputs = generate_mesh_panels(args.input, args.output_dir, pdf=args.pdf,
                                   export_data=args.export_data, verify_embeddings=args.verify_embeddings,
                                   combined=args.combined)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
