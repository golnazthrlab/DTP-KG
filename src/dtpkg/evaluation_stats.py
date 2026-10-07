"""Paired, approximate corrected-CV inference; one source for tables and figures.

Nadeau--Bengio correction: SE² = (1/n + mean(n_test/n_development)) s².
This corrects training overlap approximately, NOT shared-drug/network dependence.
It conditions on the declared cohort/graph and supplemental positive reservoir.
No inference is manufactured for old tables without split metadata, fixed
inductive holdouts repeated over seeds, undefined cells, or unpaired rows.
Pointwise CIs accompany raw two-sided p-values; Holm-adjusted p-values drive stars.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

METRICS = ("auc", "f1", "prec", "rec", "acc")
ALIASES = {"auroc": "auc", "accuracy": "acc", "precision": "prec", "recall": "rec",
           "auprc": "ap", "average_precision": "ap"}
KEYS = ["split_repeat", "fold"]
META = ["inference_protocol", "n_outer_train", "n_outer_test"]


def normalize_results(frame):
    df = frame.rename(columns={c: c.lower() for c in frame.columns}).copy()
    df = df.rename(columns={c: ALIASES[c] for c in df if c in ALIASES and ALIASES[c] not in df})
    if "category" not in df:
        for c in ("bin", "pair_bin"):
            if c in df:
                df = df.rename(columns={c: "category"})
                break
    if "category" not in df:
        df["category"] = "overall"
    if "model" not in df:
        raise ValueError("results require an explicit model column")
    return df


def corrected_interval(values, ratios, confidence=0.95, null=0.0):
    """Pointwise two-sided t interval/test using the SAME corrected SE.

    Undefined fold values invalidate inference, rather than selecting convenient
    folds. Constant nonzero contrasts cannot estimate uncertainty and get no p.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    x, ratios = np.asarray(values, float), np.asarray(ratios, float)
    if x.ndim != 1 or ratios.shape != x.shape:
        raise ValueError("values and ratios must be aligned one-dimensional arrays")
    finite = x[np.isfinite(x)]
    out = dict(estimate=float(finite.mean()) if len(finite) else np.nan,
               ci_low=np.nan, ci_high=np.nan, se=np.nan, p_raw=np.nan,
               n_folds=len(x), df=max(len(x)-1, 0), confidence=confidence,
               interval_kind="pointwise_approximate_corrected_cv",
               status="insufficient_folds")
    if not np.isfinite(x).all():
        out["status"] = "undefined_fold_metric"
        return out
    if len(x) < 2:
        return out
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("outer test/development ratios must be finite and positive")
    variance = x.var(ddof=1)
    if variance == 0:
        if np.all(x == null):
            out.update(ci_low=null, ci_high=null, se=0.0, p_raw=1.0,
                       status="identical_zero_contrast")
        else:
            out["status"] = "zero_variance_no_inference"
        return out
    se = np.sqrt((1 / len(x) + ratios.mean()) * variance)
    half = student_t.ppf((1 + confidence) / 2, len(x)-1) * se
    out.update(ci_low=out["estimate"]-half, ci_high=out["estimate"]+half, se=se,
               p_raw=float(2 * student_t.sf(abs((out["estimate"]-null)/se), len(x)-1)),
               status="ok_approximate_cv")
    return out


def holm(pvalues):
    """Preserve all declared hypotheses in the family; unavailable p counts as 1."""
    p = np.asarray(pvalues, float)
    if np.any((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)):
        raise ValueError("p-values must be in [0,1]")
    safe = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(safe, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(safe[order] * np.arange(len(p), 0, -1)))
    out = np.empty(len(p)); out[order] = adjusted
    out[~np.isfinite(p)] = np.nan
    return out


def stars(p):
    return "" if not np.isfinite(p) else "**" if p < .01 else "*" if p < .05 else ""


def _collapse_seeds(sub, metric):
    """A split/fold is one unit. Average explicitly identified training seeds."""
    if not set(KEYS + META).issubset(sub):
        return None, "missing_cv_metadata"
    if set(sub.inference_protocol) != {"cv"}:
        return None, "unsupported_protocol_no_population_inference"
    if sub[KEYS + META].isna().any().any():
        raise ValueError("null split metadata")
    if sub.duplicated(KEYS).any():
        if "training_seed" not in sub or sub.duplicated(KEYS + ["training_seed"]).any():
            raise ValueError("duplicate fold rows without unique training_seed")
    for _, g in sub.groupby(KEYS):
        if any(g[c].nunique() != 1 for c in META):
            raise ValueError("conflicting metadata for the same fold")
    # Do not let pandas silently drop undefined seed results.
    def mean_complete(x):
        return float(np.mean(x.to_numpy(float)))
    cols = sub.groupby(KEYS, sort=True).agg({metric: mean_complete,
                                           **{c: "first" for c in META}})
    for c in ("n_outer_train", "n_outer_test"):
        if (cols[c] <= 0).any():
            raise ValueError("outer partition sizes must be positive")
    return cols, None


def performance_summary(frame, metrics=METRICS, confidence=.95):
    """Per-model fold means and pointwise approximate CIs; SD is always descriptive."""
    df = normalize_results(frame)
    rows = []
    for (cat, model), sub in df.groupby(["category", "model"], sort=False):
        for metric in metrics:
            if metric not in sub:
                continue
            collapsed, reason = _collapse_seeds(sub, metric)
            if "status" in sub and sub.status.eq("not_applicable_no_ddi_edges").any():
                if not sub.status.eq("not_applicable_no_ddi_edges").all() or sub[metric].notna().any():
                    raise ValueError("Not-applicable graph baseline rows must consistently have blank metrics")
                collapsed, reason = None, "not_applicable_no_ddi_edges"
            values = sub[metric].to_numpy(float) if collapsed is None else collapsed[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            if collapsed is None:
                res = dict(estimate=float(finite.mean()) if len(finite) else np.nan,
                           ci_low=np.nan, ci_high=np.nan, status=reason,
                           interval_kind="unavailable", confidence=confidence, n_folds=0)
            else:
                res = corrected_interval(values, (collapsed.n_outer_test / collapsed.n_outer_train).values,
                                         confidence=confidence)
            res.pop("p_raw", None)
            rows.append(dict(category=cat, model=model, metric=metric,
                             sd=float(finite.std(ddof=1)) if len(finite)>1 else np.nan, **res))
    return pd.DataFrame(rows)


def comparison_table(frame, contrasts=(("fusion", "baseline"),), metrics=METRICS,
                     confidence=.95, family="category_comparisons", categories=None):
    """Strictly paired differences (first model minus second); Holm across the
    entire supplied category × metric × contrast family, not per plot panel.
    Pass contrasts='all' for each unique two-sided comparison once.
    """
    df = normalize_results(frame)
    if contrasts == "all":
        contrasts = list(combinations(sorted(df.model.unique()), 2))
    contrasts = list(contrasts)
    identities = [frozenset(c) for c in contrasts]
    if any(len(c) != 2 for c in identities) or len(set(identities)) != len(identities):
        raise ValueError("contrasts must contain distinct, unique unordered model pairs")
    cats = list(df.category.unique()) if categories is None else list(categories)
    rows = []
    for cat in cats:
        for a, b in contrasts:
            for metric in metrics:
                row = dict(category=cat, model_a=a, model_b=b, metric=metric, family=family,
                           estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p_raw=np.nan,
                           status="missing_model_or_metric", n_folds=0,
                           confidence=confidence, interval_kind="unavailable")
                aa = df[(df.category == cat) & (df.model == a)]
                bb = df[(df.category == cat) & (df.model == b)]
                if metric not in df or aa.empty or bb.empty:
                    rows.append(row); continue
                av, ar = _collapse_seeds(aa, metric)
                bv, br = _collapse_seeds(bb, metric)
                if ar or br:
                    row["status"] = ar or br
                    # Descriptive difference only; never paired by row order.
                    row["estimate"] = aa[metric].mean() - bb[metric].mean()
                elif not av.index.equals(bv.index):
                    raise ValueError(f"unmatched split/fold keys for {a} vs {b}, {cat}")
                elif not av[META].equals(bv[META]):
                    raise ValueError("paired methods have different outer partition metadata")
                else:
                    if "training_seed" in aa and "training_seed" in bb:
                        ka = set(map(tuple, aa[KEYS+["training_seed"]].values))
                        kb = set(map(tuple, bb[KEYS+["training_seed"]].values))
                        if ka != kb:
                            raise ValueError("paired methods have different training seeds")
                    row.update(corrected_interval((av[metric]-bv[metric]).values,
                               (av.n_outer_test / av.n_outer_train).values, confidence))
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out.p_raw)
    out["family_size"] = len(out)
    out["stars"] = out.p_holm.map(stars)
    out["direction"] = np.where(out.estimate > 0, "a_higher", np.where(out.estimate < 0, "b_higher", "equal"))
    return out
