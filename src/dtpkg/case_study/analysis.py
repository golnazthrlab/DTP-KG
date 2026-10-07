"""Descriptive protein-overlap analysis on the graphs used for prediction.

Radius is measured from the drug, not its targets. Protein membership includes
both ``target`` and ``protein`` nodes. Full-graph paths may pass through another
drug; the biological-path sensitivity excludes such DDI-mediated paths.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


PROTEIN_TYPES = frozenset(("target", "protein"))
FEATURES = ("shared_protein_count", "protein_jaccard", "shared_target_count",
            "biological_shared_protein_count", "protein_size_geomean")
FIT_META = ("setting", "split_id", "split_repeat", "training_seed", "fold")
ADJUSTMENTS = {
    "shared_protein_count": ("n_proteins_a", "n_proteins_b"),
    "biological_shared_protein_count": ("n_biological_proteins_a", "n_biological_proteins_b"),
    "shared_target_count": ("n_targets_a", "n_targets_b"),
}
EXAMPLE_SOURCE = {"transductive": {"split_repeat": 0, "fold": 0},
                  "inductive": {"split_id": "split_00", "training_seed": 101}}


class _Neighborhoods:
    def __init__(self, graph):
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("case studies require a simple undirected graph")
        self.graph = graph
        self.proteins = {n for n, d in graph.nodes(data=True)
                         if str(d.get("type", "")).lower() in PROTEIN_TYPES}
        self.adjacency = {}
        self.cache = {}

    def direct(self, node):
        if node not in self.adjacency:
            self.adjacency[node] = self.proteins.intersection(self.graph[node])
        return self.adjacency[node]

    def for_drug(self, drug):
        if drug not in self.graph or str(self.graph.nodes[drug].get("type", "")).lower() != "drug":
            raise ValueError(f"missing drug or incorrect node type in endpoint graph: {drug}")
        if drug not in self.cache:
            targets = self.direct(drug)
            full, biological = set(targets), set(targets)
            for node in self.graph[drug]:
                full.update(self.direct(node))
            for node in targets:
                biological.update(self.direct(node))
            self.cache[drug] = (targets, full, biological)
        return self.cache[drug]


def protein_neighborhood(graph, drug):
    """All protein/target nodes at distance <= 2 from this drug."""
    return _Neighborhoods(graph).for_drug(drug)[1]


def biological_neighborhood(graph, drug):
    """Direct targets and one protein-to-protein step, excluding DDI paths."""
    return _Neighborhoods(graph).for_drug(drug)[2]


def annotate_pairs(predictions, train_graph, inference_graph=None, heldout_drugs=()):
    """Annotate every prediction without dropping zero-target/zero-overlap pairs.

    Seen endpoints use training graphs; held-out endpoints use inference graphs.
    A missing drug fails rather than being silently treated as an isolate.
    """
    required = {"drug1", "drug2", "label", "score", "threshold"}
    if not required.issubset(predictions):
        raise ValueError(f"missing prediction columns: {sorted(required - set(predictions))}")
    p = predictions.copy()
    if p.empty or p[list(required)].isna().any().any():
        raise ValueError("nonempty, complete predictions required")
    if not p.label.isin((0, 1)).all():
        raise ValueError("reference label must be 0 or 1")
    numbers = p[["score", "threshold"]].to_numpy(float)
    if not np.isfinite(numbers).all() or ((numbers < 0) | (numbers > 1)).any():
        raise ValueError("scores and thresholds must be finite probabilities in [0, 1]")
    a, b = p.drug1.astype(str), p.drug2.astype(str)
    if a.eq(b).any():
        raise ValueError("self pairs are not supported")
    p["pair_id"] = np.minimum(a, b) + "|" + np.maximum(a, b)
    keys = (["fit_id"] if "fit_id" in p else []) + ["pair_id"]
    if p.duplicated(keys).any():
        raise ValueError("duplicate unordered pair in a fit")
    heldout = set(heldout_drugs)
    if heldout and inference_graph is None:
        raise ValueError("held-out drugs require their inference graph")
    train = _Neighborhoods(train_graph)
    infer = (train if inference_graph is None or inference_graph is train_graph
             else _Neighborhoods(inference_graph))
    rows = []
    for row in p.itertuples(index=False):
        ha, hb = row.drug1 in heldout, row.drug2 in heldout
        ta, pa, ba = (infer if ha else train).for_drug(row.drug1)
        tb, pb, bb = (infer if hb else train).for_drug(row.drug2)
        overlap, union = len(pa & pb), len(pa | pb)
        rows.append(dict(shared_protein_count=overlap, protein_union_size=union,
            protein_jaccard=overlap / union if union else np.nan,
            shared_target_count=len(ta & tb), biological_shared_protein_count=len(ba & bb),
            n_proteins_a=len(pa), n_proteins_b=len(pb), n_targets_a=len(ta), n_targets_b=len(tb),
            n_biological_proteins_a=len(ba), n_biological_proteins_b=len(bb),
            protein_size_geomean=np.sqrt(len(pa) * len(pb)),
            drug1_heldout=ha, drug2_heldout=hb,
            endpoint_graph_a="inference" if ha else "training",
            endpoint_graph_b="inference" if hb else "training"))
    # Replace any saved endpoint flags only after verifying that they agree.
    annotations = pd.DataFrame(rows, index=p.index)
    for name in ("drug1_heldout", "drug2_heldout"):
        if name in p and not p[name].eq(annotations[name]).all():
            raise ValueError("saved held-out flags disagree with the source split")
    for name in annotations:
        p[name] = annotations[name]
    return p


def correlations(table):
    """Within-fit Spearman coefficients; no independence-based p-values or CIs.

    Undefined Jaccard for two empty neighborhoods is excluded only for that
    feature and its count is reported. Other constant inputs give an explicit
    undefined result, never a manufactured zero correlation.
    """
    rows = []
    for (fit_id, scenario), group in table.groupby(["fit_id", "scenario"], sort=True):
        metadata = {}
        for name in FIT_META:
            if name in group:
                if group[name].nunique(dropna=False) != 1:
                    raise ValueError(f"conflicting fit metadata: {name}")
                metadata[name] = group[name].iloc[0]
        for subset in ("all_pairs", "both_have_targets"):
            available = (group if subset == "all_pairs" else
                         group[(group.n_targets_a > 0) & (group.n_targets_b > 0)])
            for label_group, label in (("overall", None), ("positive", 1), ("negative", 0)):
                sub = available if label is None else available[available.label == label]
                for feature in FEATURES:
                    if feature not in sub:
                        continue
                    valid = np.isfinite(sub[feature]) & np.isfinite(sub.score)
                    finite = sub.loc[valid]
                    status, rho = "ok", np.nan
                    if len(finite) < 3:
                        status = "insufficient_pairs"
                    elif finite[feature].nunique() < 2:
                        status = "constant_feature"
                    elif finite.score.nunique() < 2:
                        status = "constant_score"
                    else:
                        rho = float(spearmanr(finite[feature], finite.score).statistic)
                    rows.append(dict(**metadata, fit_id=fit_id, scenario=scenario,
                        subset=subset, label_group=label_group, feature=feature, rho=rho,
                        statistic="spearman", controls="none", control_encoding="none",
                        n=len(finite), n_total=len(sub), n_undefined_feature=int((~valid).sum()),
                        n_unique_pairs=finite.pair_id.nunique(),
                        n_drugs=len(set(finite.drug1) | set(finite.drug2)),
                        n_unique_scores=finite.score.nunique(),
                        n_zero_overlap=int(finite.shared_protein_count.eq(0).sum()),
                        n_saturated_scores=int(finite.score.isin((0., 1.)).sum()),
                        n_total_zero_overlap=int(sub.shared_protein_count.eq(0).sum()),
                        n_total_saturated_scores=int(sub.score.isin((0., 1.)).sum()), status=status))
    return pd.DataFrame(rows)


def rank_residual_spearman(x, y, controls):
    """Exploratory Pearson correlation of rank residuals, including an intercept.

    Rank each column with average ties, regress ranked x and y separately on
    ranked controls, then correlate the residuals (do not rank them again).
    Effective design rank handles constant/collinear controls. This supplies no
    causal adjustment, p-value, or independence-based interval. ``residual_df``
    is the OLS residual-space dimension n - rank(design), where design includes
    the intercept. It is not the degrees of freedom of a correlation test.
    At least two residual dimensions are required: in a one-dimensional space,
    any two nonzero residual vectors necessarily correlate at +1 or -1.
    """
    x, y, controls = np.asarray(x, float), np.asarray(y, float), np.asarray(controls, float)
    if controls.ndim == 1:
        controls = controls[:, None]
    if x.ndim != 1 or y.shape != x.shape or controls.ndim != 2 or len(controls) != len(x):
        raise ValueError("aligned vectors and a two-dimensional control matrix required")
    if not all(np.isfinite(v).all() for v in (x, y, controls)):
        raise ValueError("rank residual inputs must be finite")
    out = dict(rho=np.nan, status="insufficient_pairs", design_rank=0, residual_df=0)
    if not len(x):
        return out
    ranked = rankdata(np.column_stack([x, y]), method="average", axis=0)
    ranked_controls = rankdata(controls, method="average", axis=0)
    # Center/scale for stable rank and residual calculations; the intercept is
    # retained. Constant columns become zero, so their effective rank is zero.
    ranked_controls -= ranked_controls.mean(axis=0)
    scale = np.linalg.norm(ranked_controls, axis=0)
    ranked_controls /= np.where(scale > 0, scale, 1)
    design = np.column_stack([np.ones(len(x)), ranked_controls])
    beta, _, rank, _ = np.linalg.lstsq(design, ranked, rcond=None)
    out.update(design_rank=int(rank), residual_df=len(x) - int(rank))
    if len(x) < 3:
        return out
    if np.unique(x).size < 2:
        out["status"] = "constant_feature"
    elif np.unique(y).size < 2:
        out["status"] = "constant_score"
    elif out["residual_df"] < 2:
        out["status"] = "insufficient_residual_df"
    else:
        residual = ranked - design @ beta
        residual -= residual.mean(axis=0)
        norms = np.linalg.norm(residual, axis=0)
        tolerances = 1e-10 * np.maximum(1, np.linalg.norm(ranked - ranked.mean(axis=0), axis=0))
        if norms[0] <= tolerances[0]:
            out["status"] = "constant_feature_residual"
        elif norms[1] <= tolerances[1]:
            out["status"] = "constant_score_residual"
        else:
            out.update(rho=float(np.clip(np.dot(residual[:, 0] / norms[0], residual[:, 1] / norms[1]), -1, 1)),
                       status="ok")
    return out


def partial_correlations(table):
    """Size-adjusted exploratory rows under two explicitly separate encodings.

    ``endpoint_columns`` reproduces adjustment for the saved a/b size columns.
    ``unordered_minmax`` sorts raw endpoint sizes within each pair *before*
    ranking and is invariant to swapping either drug pair. It is a separate
    model-specification sensitivity, not a substitute selected for a larger rho.
    """
    rows = []
    for (fit_id, scenario), group in table.groupby(["fit_id", "scenario"], sort=True):
        metadata = {name: group[name].iloc[0] for name in FIT_META if name in group}
        if any(group[name].nunique(dropna=False) != 1 for name in metadata):
            raise ValueError("conflicting fit metadata")
        for subset in ("all_pairs", "both_have_targets"):
            available = group if subset == "all_pairs" else group[(group.n_targets_a > 0) & (group.n_targets_b > 0)]
            for label_group, label in (("overall", None), ("positive", 1), ("negative", 0)):
                sub = available if label is None else available[available.label == label]
                for feature, control_names in ADJUSTMENTS.items():
                    if not {feature, *control_names}.issubset(sub):
                        continue
                    valid = np.isfinite(sub[[feature, "score", *control_names]].to_numpy(float)).all(axis=1)
                    finite = sub.loc[valid]
                    for encoding in ("endpoint_columns", "unordered_minmax"):
                        controls = finite[list(control_names)].to_numpy(float)
                        if encoding == "unordered_minmax":
                            controls = np.sort(controls, axis=1)
                        result = rank_residual_spearman(finite[feature], finite.score, controls)
                        rows.append(dict(**metadata, fit_id=fit_id, scenario=scenario,
                            subset=subset, label_group=label_group, feature=feature,
                            statistic="partial_spearman", controls="|".join(control_names),
                            control_encoding=encoding, **result,
                            n=len(finite), n_total=len(sub), n_undefined_feature=int((~valid).sum()),
                            n_unique_pairs=finite.pair_id.nunique(),
                            n_drugs=len(set(finite.drug1) | set(finite.drug2)),
                            n_unique_scores=finite.score.nunique(),
                            n_zero_overlap=int(finite.shared_protein_count.eq(0).sum()),
                            n_saturated_scores=int(finite.score.isin((0., 1.)).sum()),
                            n_total_zero_overlap=int(sub.shared_protein_count.eq(0).sum()),
                            n_total_saturated_scores=int(sub.score.isin((0., 1.)).sum())))
    return pd.DataFrame(rows)


def coverage_partial_correlations(table):
    """Exploratory biological-overlap adjustment with a zero-size indicator.

    Report all pairs and both reference classes together, separately for each
    fit/scenario. Append ``either_biological_size_zero`` to each existing size
    encoding. This indicator is 1 exactly when either biological-neighborhood
    size equals zero. Ranking a binary indicator is affine-equivalent to using
    it directly when an intercept is present, so the shared rank-residual
    routine preserves the intended binary adjustment.

    Ranking minimum size does not in general span a separate step at zero;
    adding this control is a distinct post hoc coverage diagnostic. Differences
    from the size-only adjustments do not identify a causal coverage effect or
    make an all-pairs coefficient equivalent to a target-covered-subset result.
    """
    feature = "biological_shared_protein_count"
    size_names = ADJUSTMENTS[feature]
    indicator_name = "either_biological_size_zero"
    rows = []
    for (fit_id, scenario), group in table.groupby(["fit_id", "scenario"], sort=True):
        metadata = {name: group[name].iloc[0] for name in FIT_META if name in group}
        if any(group[name].nunique(dropna=False) != 1 for name in metadata):
            raise ValueError("conflicting fit metadata")
        if not {feature, *size_names}.issubset(group):
            continue
        valid = np.isfinite(group[[feature, "score", *size_names]].to_numpy(float)).all(axis=1)
        finite = group.loc[valid]
        sizes = finite[list(size_names)].to_numpy(float)
        indicator = (sizes == 0).any(axis=1).astype(float)
        for encoding in ("endpoint_columns_plus_zero", "unordered_minmax_plus_zero"):
            encoded_sizes = np.sort(sizes, axis=1) if encoding.startswith("unordered_minmax") else sizes
            controls = np.column_stack([encoded_sizes, indicator])
            result = rank_residual_spearman(finite[feature], finite.score, controls)
            rows.append(dict(**metadata, fit_id=fit_id, scenario=scenario,
                subset="all_pairs", label_group="overall", feature=feature,
                statistic="partial_spearman", controls="|".join((*size_names, indicator_name)),
                control_encoding=encoding, **result,
                n=len(finite), n_total=len(group), n_undefined_feature=int((~valid).sum()),
                n_unique_pairs=finite.pair_id.nunique(),
                n_drugs=len(set(finite.drug1) | set(finite.drug2)),
                n_unique_scores=finite.score.nunique(),
                n_zero_overlap=int(finite.shared_protein_count.eq(0).sum()),
                n_saturated_scores=int(finite.score.isin((0., 1.)).sum()),
                n_total_zero_overlap=int(group.shared_protein_count.eq(0).sum()),
                n_total_saturated_scores=int(group.score.isin((0., 1.)).sum())))
    return pd.DataFrame(rows)


def _complete_mean(values):
    x = np.asarray(values, dtype=float)
    return float(x.mean()) if len(x) and np.isfinite(x).all() else np.nan


def summarize_correlations(frame):
    """Equal weight per repeat/holdout, after averaging training seeds and folds.

    Means/SD/ranges describe variability; they are not intervals or significance
    tests. Any undefined constituent propagates to the summary estimate.
    """
    frame = frame.copy()
    if "subset" not in frame:
        frame["subset"] = "all_pairs"
    for name, default in (("statistic", "spearman"), ("controls", "none"), ("control_encoding", "none")):
        if name not in frame:
            frame[name] = default
    groups = ["setting", "scenario", "subset", "label_group", "feature",
              "statistic", "controls", "control_encoding"]
    splits = []
    for keys, sub in frame.groupby(groups + ["split_repeat"], sort=True):
        meta = dict(zip(groups + ["split_repeat"], keys))
        fold_means = sub.groupby("fold", dropna=False).rho.agg(_complete_mean)
        mean = _complete_mean(fold_means)
        splits.append(dict(**meta, mean_rho=mean, n_fits=len(sub),
                           n_min=int(sub.n.min()) if "n" in sub else np.nan,
                           n_max=int(sub.n.max()) if "n" in sub else np.nan,
                           n_defined_fits=int(sub.rho.notna().sum()),
                           status="ok" if np.isfinite(mean) else "undefined_fit"))
    split_table = pd.DataFrame(splits)
    rows = []
    for keys, sub in split_table.groupby(groups, sort=True):
        x = sub.mean_rho.to_numpy(float)
        complete = np.isfinite(x).all()
        rows.append(dict(zip(groups, keys), mean_rho=_complete_mean(x),
            sd_rho=float(np.std(x, ddof=1)) if complete and len(x) > 1 else np.nan,
            min_rho=float(np.min(x)) if complete else np.nan,
            max_rho=float(np.max(x)) if complete else np.nan,
            n_splits=len(x), n_defined_splits=int(np.isfinite(x).sum()),
            n_fits=int(sub.n_fits.sum()), status="ok" if complete else "undefined_split",
            n_min=sub.n_min.min(), n_max=sub.n_max.max(),
            interpretation="descriptive; SD and range are not confidence intervals"))
    return split_table, pd.DataFrame(rows)


def prespecified_example_mask(table):
    """Single source of truth for the fixed example-fit choice."""
    mask = pd.Series(False, index=table.index)
    for setting in table.setting.unique():
        if setting not in EXAMPLE_SOURCE:
            raise ValueError(f"unknown setting: {setting}")
        selected = table.setting.eq(setting)
        for column, value in EXAMPLE_SOURCE[setting].items():
            selected &= table[column].eq(value)
        mask |= selected
    return mask


def select_examples(table, positive_cutoff=.9, negative_cutoff=.1):
    """Select extremes in the first prespecified fit, never by overlap.

    Both drugs must have recorded targets (figure eligibility only). A missing
    qualifying example is reported; neither cutoffs nor source fits are relaxed.
    """
    if not 0 <= negative_cutoff < .5 < positive_cutoff <= 1:
        raise ValueError("cutoffs must bound .5 within [0, 1]")
    selected, logs = [], []
    for (setting, scenario), group in table.groupby(["setting", "scenario"], sort=True):
        first = group[prespecified_example_mask(group)]
        if first.fit_id.nunique() > 1:
            raise ValueError("prespecified example source identifies more than one fit")
        eligible = first[(first.n_targets_a > 0) & (first.n_targets_b > 0)]
        for kind, label, cutoff in (("high_score_positive", 1, positive_cutoff),
                                    ("low_score_reliable_negative", 0, negative_cutoff)):
            candidates = eligible[eligible.label == label]
            candidates = (candidates[(candidates.score >= cutoff) & (candidates.score >= candidates.threshold)]
                if label else candidates[(candidates.score <= cutoff) & (candidates.score < candidates.threshold)])
            candidates = candidates.sort_values(["score", "pair_id"], ascending=[not label, True], kind="stable")
            tied = int(candidates.score.eq(candidates.score.iloc[0]).sum()) if len(candidates) else 0
            excluded = first[(first.label == label) & ((first.n_targets_a == 0) | (first.n_targets_b == 0))]
            excluded = (excluded[(excluded.score >= cutoff) & (excluded.score >= excluded.threshold)] if label
                        else excluded[(excluded.score <= cutoff) & (excluded.score < excluded.threshold)])
            status = "source_fit_absent" if first.empty else "selected" if len(candidates) else "no_qualifying_pair"
            logs.append(dict(setting=setting, scenario=scenario, example_kind=kind,
                cutoff=cutoff, n_source_pairs=len(first), n_target_eligible=len(eligible),
                n_candidates=len(candidates), n_tied_extreme=tied,
                n_excluded_targetless_qualifying=len(excluded), status=status,
                rule="first prespecified fit; both targets; correct reference class; score then pair ID"))
            if len(candidates):
                row = candidates.iloc[0].to_dict()
                row["example_kind"] = kind
                row["selection_tie_count"] = tied
                row["n_excluded_targetless_qualifying"] = len(excluded)
                selected.append(row)
    return pd.DataFrame(selected), pd.DataFrame(logs)
