"""Report saved ablation fits; never load a model or the original input files.

Fold means, paired corrected-CV intervals, and prespecified Holm families use
the repository's shared inference implementation. Incomplete studies are
descriptive only. Threshold selection always uses inner-validation scores.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dtpkg.ablation.artifacts import digest, file_hash, validate_completed, write_json
from dtpkg.ddi_sampling import epoch_seed
from dtpkg.evaluation_stats import comparison_table, performance_summary
from dtpkg.fusion.graph_baselines import validation_f1_threshold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, matthews_corrcoef)
from dtpkg.metrics import PAIR_BINS, _bin_result_dict

IDENTITY = ("protocol_id", "data_identity", "split_manifest_hash")
FIT_KEYS = ("model", "split_repeat", "fold", "training_seed")
META = ("inference_protocol", "n_outer_train", "n_outer_test")
ANNOTATIONS = ("has_target", "in_training_negative_roster", "isolated")
CATEGORIES = ("overall", *PAIR_BINS)



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
    out = dict.fromkeys(("auc", "ap", "f1", "prec", "rec", "specificity", "balanced_acc", "mcc", "acc"), np.nan)
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


def _read_json(path):
    return json.loads(Path(path).read_text())


def _canonical(frame):
    """Normalize endpoint order, including endpoint annotations, before pairing."""
    frame = frame.copy()
    if frame[["drug1", "drug2", "label"]].isna().any().any():
        raise ValueError("Missing pair identities/labels")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Pair labels must be binary")
    frame[["drug1", "drug2"]] = frame[["drug1", "drug2"]].astype(str)
    flip = frame.drug1 > frame.drug2
    for left, right in [("drug1", "drug2"), *[(f"{name}1", f"{name}2") for name in ANNOTATIONS]]:
        if left in frame and right in frame:
            values = frame.loc[flip, [right, left]].to_numpy()
            frame.loc[flip, [left, right]] = values
    if frame.duplicated(["drug1", "drug2"]).any():
        raise ValueError("Duplicate unordered pair in saved predictions/manifest")
    return frame.sort_values(["drug1", "drug2"]).reset_index(drop=True)


def _require_equal(actual, expected, message):
    if actual.shape != expected.shape or not np.array_equal(actual.to_numpy(), expected.to_numpy()):
        raise ValueError(message)


def _predictions(path, metadata, manifest_pairs):
    frame = pd.read_csv(path, dtype={"drug1": str, "drug2": str})
    required = {*FIT_KEYS, "drug1", "drug2", "label", "score", "pair_bin", "threshold",
                *[f"{name}{i}" for name in ANNOTATIONS for i in (1, 2)]}
    if missing := required - set(frame):
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[["score", "threshold"]].to_numpy(float)).all():
        raise ValueError(f"{path}: empty or nonfinite predictions")
    if not frame.pair_bin.isin(PAIR_BINS).all():
        raise ValueError(f"{path}: unknown or missing depth bins")
    for column in FIT_KEYS:
        if frame[column].nunique(dropna=False) != 1 or frame[column].iloc[0] != metadata[column]:
            raise ValueError(f"{path}: conflicting fit metadata {column}")
    for column in META:
        if column in frame and (frame[column].nunique(dropna=False) != 1 or frame[column].iloc[0] != metadata[column]):
            raise ValueError(f"{path}: conflicting CV metadata {column}")
    for name in ANNOTATIONS:
        for endpoint in (1, 2):
            column = f"{name}{endpoint}"
            if column in frame:
                if frame[column].isna().any() or not frame[column].isin([True, False, 0, 1]).all():
                    raise ValueError(f"{path}: missing/invalid {column}")
                frame[column] = frame[column].astype(bool)
    frame = _canonical(frame)
    _require_equal(frame[["drug1", "drug2", "label"]], _canonical(manifest_pairs)[["drug1", "drug2", "label"]],
                   f"{path}: pair identities or labels disagree with saved split manifest")
    return frame


def _metrics(frame, metadata, threshold=None):
    rows = []
    for category in CATEGORIES:
        sub = frame if category == "overall" else frame[frame.pair_bin == category]
        score, labels = sub.score.to_numpy(float), sub.label.to_numpy(int)
        cut = sub.threshold.to_numpy(float) if threshold is None else threshold
        rows.append({**metadata, "category": category,
                     **_bin_result_dict(labels, score, (score >= cut).astype(int))})
    return pd.DataFrame(rows)


def _load(output_dir, allow_partial):
    protocol = _read_json(output_dir / "protocol.json")
    if protocol.get("protocol_id") != digest({k: v for k, v in protocol.items() if k != "protocol_id"}):
        raise ValueError("Saved protocol digest mismatch")
    manifest_path = output_dir / "split_manifest.json"
    if not manifest_path.is_file() or file_hash(manifest_path) != protocol["split_manifest_hash"]:
        raise ValueError("Saved split manifest hash mismatch")
    manifest = _read_json(manifest_path)
    pairs = pd.DataFrame(manifest["pairs"])
    _canonical(pairs)
    pair_records = [(str(a), str(b), int(label)) for a, b, label in
                    pairs[["drug1", "drug2", "label"]].itertuples(index=False, name=None)]
    if "source_files" in protocol and protocol["data_identity"] != digest({
            "files": {name: value["sha256"] for name, value in protocol["source_files"].items()},
            "pairs": pair_records}):
        raise ValueError("Saved data identity disagrees with source identities and manifest pairs")
    arms = {arm["name"]: arm for arm in protocol["arms"]}
    if len(arms) != len(protocol["arms"]):
        raise ValueError("Duplicate protocol arms")
    expected = protocol["expected_fits"]
    keys = [(e["arm"], e["split_repeat"], e["fold"]) for e in expected]
    if not keys or len(set(keys)) != len(keys) or any(key[0] not in arms for key in keys):
        raise ValueError("Missing, duplicate or unknown expected fits")
    expected_paths = {output_dir / "fits" / arm / f"repeat_{rep:02d}_fold_{fold:02d}" for arm, rep, fold in keys}
    if any(marker.parent not in expected_paths for marker in (output_dir / "fits").glob("*/repeat_*_fold_*/completed.json")):
        raise ValueError("Unexpected completed fits outside saved protocol")
    missing, fits, by_coordinate = [], [], {}
    required_files = {"fold_metrics.csv", "predictions.csv", "validation_predictions.csv", "fit_info.json", "history.csv"}
    for arm, rep, fold in keys:
        folder = output_dir / "fits" / arm / f"repeat_{rep:02d}_fold_{fold:02d}"
        identity = {key: protocol[key] for key in IDENTITY} | dict(arm=arm, split_repeat=rep, fold=fold)
        if not validate_completed(folder, identity):
            missing.append(dict(arm=arm, split_repeat=rep, fold=fold))
            continue
        marker = _read_json(folder / "completed.json")
        if required_files - set(marker["artifact_hashes"]):
            raise ValueError(f"{folder}: required artifacts missing from completion record")
        if any(Path(name).name != name for name in marker["artifact_hashes"]):
            raise ValueError("Artifact paths must be local filenames")
        metrics = pd.read_csv(folder / "fold_metrics.csv")
        if set(metrics.category) != set(CATEGORIES) or metrics.category.duplicated().any():
            raise ValueError(f"{folder}: require exactly one row for all seven categories")
        columns = [*FIT_KEYS, *META]
        if not set(columns).issubset(metrics) or metrics[columns].isna().any().any():
            raise ValueError(f"{folder}: missing fold metadata")
        if any(metrics[column].nunique() != 1 for column in columns):
            raise ValueError(f"{folder}: inconsistent fold metadata")
        metadata = metrics.iloc[0][columns].to_dict()
        if (metadata["model"], metadata["split_repeat"], metadata["fold"]) != (arm, rep, fold):
            raise ValueError(f"{folder}: wrong saved fit identity")
        if metadata["training_seed"] != epoch_seed(protocol["scientific_config"]["seed"], fold, rep, 0):
            raise ValueError(f"{folder}: training seed disagrees with protocol")
        assignment = manifest["repetitions"][rep][fold]
        partitions = {name: pairs.iloc[assignment[f"{name}_idx"]] for name in ("train", "val", "test")}
        identities = {name: set(map(tuple, _canonical(frame)[["drug1", "drug2"]].to_numpy()))
                      for name, frame in partitions.items()}
        if any(identities[a] & identities[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("Saved manifest has overlapping train/validation/test pairs")
        if (metadata["inference_protocol"] != "cv" or metadata["n_outer_test"] != len(partitions["test"])
                or metadata["n_outer_train"] != len(partitions["train"]) + len(partitions["val"])):
            raise ValueError("Fold metadata disagrees with saved manifest")
        test = _predictions(folder / "predictions.csv", metadata, partitions["test"])
        validation = _predictions(folder / "validation_predictions.csv", metadata, partitions["val"])
        if arms[arm].get("model_kind") in ("fusion", "mesh", "topo_only"):
            if not test.threshold.eq(.5).all() or not validation.threshold.eq(.5).all():
                raise ValueError("Primary neural predictions must use threshold 0.5")
        else:
            cutoff = validation_f1_threshold(validation.label, validation.score)
            if not test.threshold.eq(cutoff).all() or not validation.threshold.eq(cutoff).all():
                raise ValueError("Diagnostic primary threshold disagrees with inner-validation selection")
        negatives = partitions["train"][partitions["train"].label == 0]
        roster = set(negatives.drug1.astype(str)) | set(negatives.drug2.astype(str))
        for predictions in (test, validation):
            for endpoint in (1, 2):
                if not np.array_equal(predictions[f"drug{endpoint}"].isin(roster),
                                      predictions[f"in_training_negative_roster{endpoint}"]):
                    raise ValueError("Training-negative roster disagrees with inner-training manifest")
        observed = _metrics(test, metadata).set_index("category")
        saved = metrics.set_index("category").reindex(observed.index)
        for column in observed.columns.difference(columns):
            if column not in saved or not np.allclose(saved[column], observed[column], equal_nan=True, atol=1e-7, rtol=1e-7):
                raise ValueError(f"{folder}: saved fold metric {column} disagrees with predictions")
        pairing = ["drug1", "drug2", "label", "pair_bin", "training_seed"]
        pairing += [f"{name}{i}" for name in ANNOTATIONS[:2] for i in (1, 2) if f"{name}{i}" in test]
        for phase, frame in (("test", test), ("validation", validation)):
            coordinate = (rep, fold, phase)
            matched = frame[pairing]
            if coordinate in by_coordinate:
                _require_equal(matched, by_coordinate[coordinate], "Unmatched paired IDs, labels, bins, seeds or annotations across arms")
            else:
                by_coordinate[coordinate] = matched
        fits.append(dict(metadata=metadata, metrics=metrics, test=test, validation=validation,
                         info=_read_json(folder / "fit_info.json"), folder=folder))
    if missing and not allow_partial:
        raise ValueError(f"Study incomplete: {len(missing)} of {len(keys)} fits missing; use allow_partial=True for descriptive output")
    return protocol, fits, missing


def _summary(frame, metrics, confidence, complete):
    if frame.empty:
        return pd.DataFrame(columns=["category", "model", "metric", "estimate", "sd", "ci_low", "ci_high", "status", "n_folds"])
    if complete:
        return performance_summary(frame, metrics=metrics, confidence=confidence)
    rows = []
    for (category, model), group in frame.groupby(["category", "model"], sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append(dict(category=category, model=model, metric=metric,
                             estimate=float(finite.mean()) if len(finite) else np.nan,
                             sd=float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                             ci_low=np.nan, ci_high=np.nan, n_folds=len(values),
                             n_defined_folds=len(finite), status="partial_descriptive_only",
                             interval_kind="unavailable", confidence=confidence))
    return pd.DataFrame(rows)


def _descriptive_subsets(fits):
    rows, isolates = [], []
    for fit in fits:
        frame, meta = fit["test"], fit["metadata"]
        for name, annotation in (("both_have_targets", "has_target"),
                                 ("both_in_training_negative_roster", "in_training_negative_roster")):
            columns = [f"{annotation}1", f"{annotation}2"]
            if not set(columns).issubset(frame):
                rows.append({**meta, "subset": name, "metric_status": "annotation_unavailable"})
                continue
            sub = frame[frame[columns].all(axis=1)]
            rows.append({**meta, "subset": name, **class_aware_metrics(sub)})
        if {"isolated1", "isolated2"}.issubset(frame):
            for label in (0, 1):
                sub = frame[frame.label == label]
                isolates.append({**meta, "label": label, "n_pairs": len(sub),
                                 "n_either_endpoint_isolated": int(sub[["isolated1", "isolated2"]].any(axis=1).sum()),
                                 "n_both_endpoints_isolated": int(sub[["isolated1", "isolated2"]].all(axis=1).sum())})
    return pd.DataFrame(rows), pd.DataFrame(isolates)


def report_study(output_dir, allow_partial=False):
    """Validate and report completed saved fits, without fitting or source data.

    ``allow_partial=True`` permits progress inspection, but suppresses all
    hypothesis tests and confidence intervals until every declared fit exists.
    CSVs and a machine-readable completion record go to ``reports/``.
    """
    output_dir = Path(output_dir)
    protocol, fits, missing = _load(output_dir, allow_partial)
    complete, confidence = not missing, protocol.get("confidence", .95)
    destination = output_dir / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([fit["metrics"] for fit in fits], ignore_index=True) if fits else pd.DataFrame()
    primary = protocol["primary_metrics"]
    category_metrics = protocol["category_metrics"]
    overall_rows = frame[frame.category == "overall"] if fits else frame
    category_rows = frame[frame.category != "overall"] if fits else frame
    overall = _summary(overall_rows, primary, confidence, complete)
    categories = _summary(category_rows, category_metrics, confidence, complete)
    empty_comparisons = pd.DataFrame(columns=["category", "model_a", "model_b", "metric", "family", "estimate",
                                             "ci_low", "ci_high", "p_raw", "p_holm", "status"])
    comparisons = empty_comparisons.copy()
    category_comparisons = empty_comparisons.copy()
    if complete:
        comparisons = comparison_table(frame, protocol["contrasts"], primary, confidence,
                                       family=f'{protocol["study"]}_overall_primary', categories=["overall"])
        category_comparisons = comparison_table(frame, protocol["contrasts"], category_metrics, confidence,
                                                family=f'{protocol["study"]}_depth_supplementary', categories=PAIR_BINS)
    thresholds, threshold_frames = [], []
    for fit in fits:
        val = fit["validation"]
        threshold = validation_f1_threshold(val.label, val.score)
        thresholds.append({**fit["metadata"], "validation_f1_threshold": threshold,
                           "selection_partition": "inner_validation"})
        threshold_frames.append(_metrics(fit["test"], fit["metadata"], threshold))
    threshold_frame = pd.concat(threshold_frames, ignore_index=True) if fits else pd.DataFrame()
    threshold_overall = threshold_frame[threshold_frame.category == "overall"] if fits else threshold_frame
    # Secondary threshold sensitivity is descriptive; primary hypotheses remain unchanged.
    threshold_summary = _summary(threshold_overall, ("f1", "prec", "rec", "acc"), confidence, complete=False)
    if complete and not threshold_summary.empty:
        threshold_summary["status"] = "secondary_descriptive_only"
    subsets, isolates = _descriptive_subsets(fits)
    exposure = pd.DataFrame([fit["info"] for fit in fits])
    gates = [pd.read_csv(fit["folder"] / "gates.csv") for fit in fits if (fit["folder"] / "gates.csv").is_file()]
    gates = [table for table in gates if not table.empty]
    gate_table = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame()
    gate_summary = pd.DataFrame()
    if not gate_table.empty:
        repeat_gates = gate_table.groupby(["model", "drug_id", "split_repeat"], as_index=False).gate_mean.mean()
        gate_summary = repeat_gates.groupby(["model", "drug_id"], as_index=False).agg(
            gate_mean=("gate_mean", "mean"), gate_sd_repeats=("gate_mean", "std"), n_repeats=("gate_mean", "size"))
    histories = []
    for fit in fits:
        history = pd.read_csv(fit["folder"] / "history.csv")
        if not history.empty:
            histories.append(history.assign(**fit["metadata"]))
    history_table = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    subset_summary = pd.DataFrame()
    if not subsets.empty and "auc" in subsets:
        subset_summary = _summary(subsets.rename(columns={"subset": "category"}),
                                  ("auc", "f1", "prec", "rec", "acc"), confidence, complete=False)
        subset_summary["status"] = "subset_descriptive_only" if complete else "partial_descriptive_only"
    status = {**{key: protocol[key] for key in IDENTITY}, "study": protocol["study"], "complete": complete,
              "expected_fits": len(protocol["expected_fits"]), "completed_fits": len(fits), "missing_fits": missing,
              "inference": "approximate_corrected_cv" if complete else "partial_descriptive_only",
              "threshold_sensitivity": "inner-validation selection; descriptive secondary analysis",
              "subset_sensitivity": "descriptive, class-aware metrics; no population inference"}
    tables = dict(overall=overall, comparisons=comparisons, categories=categories,
                  category_comparisons=category_comparisons, fold_metrics=frame,
                  validation_thresholds=pd.DataFrame(thresholds), threshold_fold_metrics=threshold_frame,
                  threshold_sensitivity=threshold_summary, subset_fold_metrics=subsets,
                  subset_summary=subset_summary, isolate_counts=isolates, fit_info=exposure,
                  gates=gate_table, gate_drug_summary=gate_summary, training_history=history_table)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    write_json(destination / "report_status.json", status)
    return {**tables, "status": status, "output_dir": destination}
