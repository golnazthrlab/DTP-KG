"""MeSH scope contrasts across test-pair categories, plotted as AUROC and F1."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dtpkg.evaluation_stats import stars


def plot_scope_heatmap(comparisons, out_dir, fname="mesh_depth_effect_heatmap", show=False):
    """Render one PNG; numerical results and captions are managed by the caller.

    Cells show the mean paired difference (first configuration minus second).
    Significance uses the supplied Holm-adjusted p-values without readjustment
    for the subset of metrics displayed here.
    """
    from dtpkg.mesh_scope.protocol import CATEGORIES, CONTRASTS
    from dtpkg.mesh_scopes import SCOPE_BY_KEY

    columns = [(metric, a, b) for metric in ("auc", "f1") for a, b in CONTRASTS]
    values = np.full((len(CATEGORIES), len(columns)), np.nan)
    labels = np.full(values.shape, "NA", dtype=object)
    for r, category in enumerate(CATEGORIES):
        for c, (metric, a, b) in enumerate(columns):
            rows = comparisons[(comparisons.metric == metric) & (comparisons.category == category)
                               & (comparisons.model_a == a) & (comparisons.model_b == b)]
            if len(rows) != 1:
                raise ValueError(f"Expected exactly one heatmap cell: {category}, {metric}, {a}, {b}")
            row = rows.iloc[0]
            values[r, c] = row.estimate
            if np.isfinite(row.estimate):
                mark = stars(row.p_holm) if np.isfinite(row.p_holm) else " (NA)"
                labels[r, c] = f"{row.estimate:+.3f}{mark}"
    finite = values[np.isfinite(values)]
    vmax = max(.005, float(np.ceil(np.abs(finite).max() / .005) * .005)) if finite.size else .005
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": True, "axes.spines.right": True}):
        fig, ax = plt.subplots(figsize=(12.8, 6.0))
        fig.subplots_adjust(left=.145, right=.90, bottom=.22, top=.88)
        im = ax.imshow(values, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                ax.text(c, r, labels[r, c], ha="center", va="center", fontsize=10.5,
                        color="white" if abs(values[r, c]) > .58 * vmax else "black")
        contrasts = [f"{SCOPE_BY_KEY[a].name}\nvs {SCOPE_BY_KEY[b].name}" for _, a, b in columns]
        ax.set_xticks(range(len(columns)), contrasts)
        ax.set_yticks(range(len(CATEGORIES)), [x.replace("-", "–").title() for x in CATEGORIES])
        for label, category in zip(ax.get_yticklabels(), CATEGORIES):
            if len(set(category.split("-"))) == 1:
                label.set_fontweight("bold")
        ax.tick_params(axis="both", length=3, pad=6)
        ax.set_xlabel("Training-scope contrast", labelpad=14)
        ax.set_ylabel("Test pair category", labelpad=10)
        ax.text(.25, 1.045, "AUROC", transform=ax.transAxes, ha="center", va="bottom", fontweight="bold", fontsize=13)
        ax.text(.75, 1.045, "F1", transform=ax.transAxes, ha="center", va="bottom", fontweight="bold", fontsize=13)
        ax.axvline(2.5, color="#a8a8a8", linestyle=":", linewidth=1)
        cax = fig.add_axes([.924, .22, .018, .66])
        fig.colorbar(im, cax=cax, label="Δ metric between training scopes")
        path = out / f"{fname}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
    return path


def load_protocol(run_dir):
    """Read an aggregate snapshot or a completed private training run.

    Public snapshots use ``protocol.json`` with ``run_plan`` and ``completion``
    objects. A private run retains the original two separate JSON files.
    """
    import json

    run = Path(run_dir)
    if (run / "protocol.json").exists():
        protocol = json.loads((run / "protocol.json").read_text())
        if protocol.get("schema_version") != 1:
            raise ValueError("Unsupported aggregate snapshot protocol version")
        plan, completion = protocol["run_plan"], protocol["completion"]
        files = ("protocol.json",)
    else:
        plan = json.loads((run / "run_plan.json").read_text())
        completion = json.loads((run / "completion.json").read_text())
        files = ("run_plan.json", "completion.json")
    from dtpkg.mesh_scope.protocol import LEVELS

    for field in ("repeats", "folds"):
        if not isinstance(plan.get(field), int) or plan[field] < 1:
            raise ValueError(f"Expected a positive integer {field} in the protocol")
    n_fits = plan["repeats"] * plan["folds"] * len(LEVELS)
    if (completion.get("status") != "complete"
            or completion.get("fits") != n_fits
            or "full training schedule" not in plan.get("scaler", "")):
        raise ValueError("Expected a completed run with the training-schedule scaler")
    return plan, completion, files


def validate_fold_grid(frame, plan, groups):
    """Require every prescribed repeat/fold once per supplied group."""
    import pandas as pd

    keys = ["split_repeat", "fold"]
    required = set(keys + list(groups))
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing scope fold results")
    if frame[list(required)].isna().any().any() or frame.duplicated(list(groups) + keys).any():
        raise ValueError("Missing or duplicate split/fold keys")
    expected = pd.MultiIndex.from_product(
        [range(plan["repeats"]), range(plan["folds"])], names=keys)
    for _, sub in frame.groupby(list(groups), sort=False):
        actual = pd.MultiIndex.from_frame(sub[keys]).sort_values()
        if not actual.equals(expected):
            raise ValueError("Missing or unmatched folds against the declared protocol")


def separate_output(run_dir, out_dir):
    """Protect the supplied tables and private result directory from writes."""
    run, out = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out == run or run in out.parents:
        raise ValueError("Figure output must be outside the input results directory")
    return run, out


def sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(run_dir, names):
    return {name: sha256(Path(run_dir) / name) for name in names}


def default_paths(run_dir=None, out_dir=None):
    from dtpkg.project_paths import EXPERIMENTS_DIR

    experiment = EXPERIMENTS_DIR / "03_mesh_scope"
    return (Path(run_dir) if run_dir is not None else experiment / "tables",
            Path(out_dir) if out_dir is not None else experiment / "figures" / "reproduced")
