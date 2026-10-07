"""Generate the distribution of deepest MeSH levels among approved mapped drugs.

Drug knowledge categories are Low 1–5, Mid 6–7, and Deep 8–10. These are
distinct from the term-depth scopes used to construct embeddings.
"""
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtpkg.project_paths import DATA_DIR, EXPERIMENTS_DIR

from dtpkg.mesh_scopes import KNOWLEDGE_CATEGORIES, load_annotations

def generate_level_distribution(annotations, output_dir, *, pdf=False, export_data=False):
    """Render deepest-level counts from an annotation DataFrame or CSV path.

    Validate stored deepest levels and categories. Return the PNG path.
    """
    # Isolate each figure from notebook themes and earlier plotting calls.
    with plt.rc_context(rc=plt.rcParamsDefault):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        OUT_PNG = output_dir / "level_dist.png"
        OUT_PDF = output_dir / "level_dist.pdf"
        ann = load_annotations(annotations) if isinstance(annotations, (str, Path)) else annotations.copy()
        deepest = ann.groupby("drugbank_id")["level"].max().astype(int)
        stored = ann.drop_duplicates("drugbank_id").set_index("drugbank_id")["deepest_level"]
        assert (deepest.sort_index() == stored.sort_index()).all(), \
            "deepest level recomputed from rows must equal the stored deepest_level column"

        # Per-level histogram
        counts = deepest.value_counts().sort_index()
        levels = np.arange(1, int(counts.index.max()) + 1)
        counts = counts.reindex(levels, fill_value=0)

        # Stratum assignment per level
        _BOUNDS = {name: (lo, hi) for name, _key, lo, hi in KNOWLEDGE_CATEGORIES}


        def stratum_of(lv: int) -> str:
            for name, (lo, hi) in _BOUNDS.items():
                if lo <= lv <= hi:
                    return name
            raise ValueError(lv)

        stratum_levels = {
            s: [lv for lv in levels if stratum_of(lv) == s]
            for s in ["Low", "Mid", "Deep"]
        }

        stratum_totals = {
            s: int(sum(counts[lv] for lv in stratum_levels[s]))
            for s in ["Low", "Mid", "Deep"]
        }

        N = sum(stratum_totals.values())

        stored_cat = ann.drop_duplicates("drugbank_id").set_index("drugbank_id")["knowledge_category"]
        recomputed = deepest.map(lambda l: {"Low": "low_level", "Mid": "mid_level", "Deep": "deep_level"}[stratum_of(l)])
        assert (stored_cat.sort_index() == recomputed.sort_index()).all(), \
            "banding must reproduce the stored knowledge_category column"

        # ── Palette + style ───────────────────────────────────────────────────────
        # Sequential blues encoding depth (light = shallow → dark = deep). Bars use the
        # ramp; headers use darker, clearly legible blues (so the "Low" label no longer
        # washes out on white).
        STRATUM_COLOR     = {"Low": "#9ecae1", "Mid": "#4292c6", "Deep": "#08519c"}
        STRATUM_TXT_COLOR = {"Low": "#2171b5", "Mid": "#2171b5", "Deep": "#08306b"}
        STRATUM_RANGE     = {name: f"{lo}–{hi}" for name, (lo, hi) in _BOUNDS.items()}

        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        })

        # ── Figure ────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        fig.patch.set_facecolor("white")

        bar_colors = [STRATUM_COLOR[stratum_of(lv)] for lv in levels]
        bars = ax.bar(
            levels,
            counts.values,
            width=0.82,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        ymax = counts.values.max()

        # Bar labels: skip very small bars if needed to avoid clutter
        for bar, v in zip(bars, counts.values):
            if v == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + ymax * 0.025,
                f"{v:,}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
                fontweight="bold",
            )

        # Stratum boundaries
        for bx in (_BOUNDS['Low'][1] + 0.5, _BOUNDS['Mid'][1] + 0.5):
            ax.axvline(
                bx,
                color="#777777",
                lw=0.8,
                ls=(0, (4, 4)),
                zorder=2,
            )

        # Compact stratum labels
        header_y = ymax * 1.20
        for s in ["Low", "Mid", "Deep"]:
            lvs = stratum_levels[s]
            if not lvs:
                continue

            mid = (min(lvs) + max(lvs)) / 2
            n = stratum_totals[s]
            pct = 100 * n / N

            ax.text(
                mid,
                header_y,
                f"{s} ({STRATUM_RANGE[s]})\n{n:,} • {pct:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=STRATUM_TXT_COLOR[s],
            )

        # Axes cleanup
        ax.set_xticks(levels)
        ax.set_xticklabels(levels)
        ax.set_xlim(0.4, levels.max() + 0.6)
        ax.set_ylim(0, ymax * 1.32)

        ax.set_xlabel("Deepest MeSH level")
        ax.set_ylabel("Number of drugs")

        ax.yaxis.grid(True, color="#e6e6e6", lw=0.7, zorder=1)
        ax.set_axisbelow(True)

        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

        plt.tight_layout()

        plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
        if pdf:
            plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
        plt.close()


        if export_data:
            data_dir = output_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"deepest_level": counts.index, "drugs": counts.values,
                          "knowledge_category": [stratum_of(int(l)) for l in counts.index]}).to_csv(
                data_dir / "level_dist_counts.csv", index=False)
        return OUT_PNG


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "mesh" / "extended_drug_info.csv")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "01_data_exploration" / "figures" / "reproduced")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--export-data", action="store_true", help="Write local aggregate plotting tables.")
    args = parser.parse_args()
    print(generate_level_distribution(args.input, args.output_dir, pdf=args.pdf, export_data=args.export_data))


if __name__ == "__main__":
    main()
