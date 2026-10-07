"""Class-aware aggregate metrics for the two inductive endpoint settings."""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, matthews_corrcoef)
from dtpkg.metrics import PAIR_BINS

ORDER = PAIR_BINS
SCENARIOS = ("seen_unseen", "unseen_unseen")
FIT_KEYS = ["split_repeat", "training_seed", "model"]
REPORT_CELLS = [("overall", "overall"), *[("depth", c) for c in ORDER]]
TYPE_METRICS = ("auc", "ap", "f1", "prec", "rec", "specificity", "balanced_acc", "mcc", "acc")


def class_aware_metrics(frame):
    """Positive-only groups: recall and positive-score summaries, NOT F1/AP/AUC.
    Thresholds are those actually used in evaluation (currently prespecified .5).
    """
    y, score = frame.label.to_numpy(int), frame.score.to_numpy(float)
    threshold = frame.threshold.to_numpy(float)
    if not np.isfinite(score).all() or not np.isfinite(threshold).all():
        raise ValueError("nonfinite predictions/thresholds")
    pred = score >= threshold
    pos, neg = y == 1, y == 0
    tp, fn = int((pred & pos).sum()), int((~pred & pos).sum())
    tn, fp = int((~pred & neg).sum()), int((pred & neg).sum())
    npos, nneg = int(pos.sum()), int(neg.sum())
    out = dict.fromkeys(TYPE_METRICS, np.nan)
    out.update(n=len(y), n_pos=npos, n_neg=nneg, tp=tp, fn=fn, tn=tn, fp=fp,
               metric_status="empty" if not len(y) else "both_classes" if npos and nneg
               else "positive_only" if npos else "negative_only")
    out["rec"] = tp/npos if npos else np.nan
    out["specificity"] = tn/nneg if nneg else np.nan
    for q, name in ((.25, "q25"), (.5, "median"), (.75, "q75")):
        out[f"positive_score_{name}"] = float(np.quantile(score[pos], q)) if npos else np.nan
    if npos and nneg:
        out.update(auc=roc_auc_score(y, score), ap=average_precision_score(y, score),
                   f1=f1_score(y, pred, zero_division=0),
                   prec=precision_score(y, pred, zero_division=0),
                   mcc=matthews_corrcoef(y, pred), acc=float((pred == y).mean()),
                   balanced_acc=(out["rec"]+out["specificity"])/2)
    return out


def single_analysis_arm(frame, what="predictions"):
    """Every reporting input belongs to exactly one declared analysis arm.

    Rows from different arms (e.g. biological-only fusion with known-context
    topology-only) would pass pair/seed matching and read as one experiment.
    A cross-arm comparison must be requested through a dedicated function, not
    by concatenating exports.
    """
    if "analysis_arm" not in frame or frame.analysis_arm.isna().any():
        raise ValueError(f"{what} must carry a non-null analysis_arm column")
    arms = sorted(frame.analysis_arm.unique())
    if len(arms) != 1:
        raise ValueError(f"{what} mix analysis arms {arms}; report each arm separately")
    return arms[0]


def scenario_metrics(predictions):
    """Aggregate private predictions into both endpoint settings and six depth cells."""
    df = predictions.copy()
    a, b = df.drug1.astype(str), df.drug2.astype(str)
    df["pair_id"] = np.where(a < b, a + "|" + b, b + "|" + a)
    if not df.label.isin([0, 1]).all() or df.groupby("pair_id").label.nunique().gt(1).any():
        raise ValueError("binary, consistent unordered pair labels required")
    arm = single_analysis_arm(df)
    identity_columns = FIT_KEYS + ["drug1", "drug2", "pair_id", "scenario", "category", "split_id",
                                  "n_heldout_drugs", "n_development_drugs"]
    if df.empty or df[identity_columns].isna().any().any():
        raise ValueError("nonempty predictions with complete fit metadata required")
    if "fit_identity" in df:
        if df.fit_identity.isna().any() or df.groupby(FIT_KEYS).fit_identity.nunique().gt(1).any():
            raise ValueError("each split/seed/model must come from exactly one identified fit")
    if not df.scenario.isin(SCENARIOS).all() or not df.category.isin(ORDER).all():
        raise ValueError("unknown inductive scenario or depth category")
    if df.duplicated(FIT_KEYS + ["pair_id"]).any():
        raise ValueError("duplicate unordered pair within a model fit")
    heldout_columns = ["drug1_heldout", "drug2_heldout"]
    if any(c in df for c in heldout_columns):
        if not all(c in df for c in heldout_columns) or not df[heldout_columns].isin([True, False]).all().all():
            raise ValueError("both endpoint held-out flags must be non-null booleans")
        nheld = df[heldout_columns].astype(int).sum(axis=1)
        if not nheld.eq(df.scenario.map({"seen_unseen": 1, "unseen_unseen": 2})).all():
            raise ValueError("held-out flags disagree with the declared scenario")
        for _, group in df.groupby("split_repeat"):
            assignments = pd.concat([group[[f"drug{i}", f"drug{i}_heldout"]].rename(
                columns={f"drug{i}": "drug", f"drug{i}_heldout": "heldout"}) for i in (1, 2)])
            if assignments.groupby("drug").heldout.nunique().gt(1).any():
                raise ValueError("inconsistent held-out assignment for a drug within a split")
    # A drug split fixes test pairs across every training seed AND every model.
    expected_fits = {(seed, model) for seed in df.training_seed.unique() for model in df.model.unique()}
    for _, group in df.groupby("split_repeat"):
        if set(map(tuple, group[["training_seed", "model"]].to_numpy())) != expected_fits:
            raise ValueError("unmatched training seeds or models across drug splits")
        if any(group[c].nunique() != 1 for c in ("split_id", "n_heldout_drugs", "n_development_drugs")):
            raise ValueError("inconsistent drug split metadata across model fits")
        reference = None
        for _, method in group.groupby(["training_seed", "model"]):
            identities = method.set_index("pair_id")[["label", "scenario", "category"]].sort_index()
            if reference is not None and not identities.equals(reference):
                raise ValueError("methods or training seeds have unmatched test pairs, labels or categories")
            reference = identities
    rows = []
    for key, group in df.groupby(FIT_KEYS, sort=True):
        metadata = dict(zip(FIT_KEYS, key))
        for col in ("n_heldout_drugs", "n_development_drugs", "split_id"):
            if group[col].nunique() != 1:
                raise ValueError("inconsistent fit metadata")
            metadata[col] = group[col].iloc[0]
        metadata["analysis_arm"] = arm
        for col in ("source_arm", "fit_identity"):
            if col in group:
                metadata[col] = group[col].iloc[0]
        for scenario in SCENARIOS:
            selected = group[group.scenario == scenario]
            cells = [(kind, category, selected if kind == "overall" else
                      selected[selected.category == category])
                     for kind, category in REPORT_CELLS]
            for group_kind, category, sub in cells:
                rows.append(dict(**metadata, scenario=scenario, group_kind=group_kind,
                                 category=category, **class_aware_metrics(sub)))
    return pd.DataFrame(rows)
