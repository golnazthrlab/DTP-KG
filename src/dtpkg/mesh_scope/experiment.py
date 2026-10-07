"""Reproducible, paired MeSH-only scope comparison for the manuscript.

The graph is not an input. Scopes share a cohort, all partitions, initialization,
and positive-sampling schedules; their fixed embeddings differ. Historical
results are never overwritten. Test contrasts are exploratory, conditional on
the common-coverage cohort and reservoir; corrected-CV inference is approximate.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

from dtpkg.mesh_scope.sampling import SamplingLoaders
from dtpkg.data_loaders import (build_pair_features, create_inner_splits,
                            load_training_positive_pool, save_split_manifest)
from dtpkg.ddi_labels import sha256
from dtpkg.evaluation_inputs import load_drug_depths, validate_drug_depths
from dtpkg.evaluation_stats import comparison_table, normalize_results, performance_summary
from dtpkg.mesh_scope.dataloader import load_paired_scope_data
from dtpkg.models.mesh_mlp import MLP_DDI
from dtpkg.mesh_scope.training import train_one_fold
from dtpkg.mesh_scopes import SCOPES
from dtpkg.project_paths import DATA_DIR
from dtpkg.metrics import evaluate_pair_scores

from dtpkg.mesh_scope.protocol import LEVELS, NAMES, CATEGORIES, CONTRASTS, METRICS


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def predict(model, loader, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in loader])


def score_pairs(pairs, scores, categories):
    bins, predictions = evaluate_pair_scores(pairs, scores, categories)
    predictions["pair_bin"] = predictions.pair_bin.str.replace("_level", "", regex=False)
    rows = []
    for name, metrics in bins.items():
        short = name.replace("_level", "")
        sub = predictions if name == "overall" else predictions[predictions.pair_bin == short]
        metrics["mcc"] = matthews_corrcoef(sub.label, sub.binary) if len(sub) else np.nan
        rows.append(dict(pair_bin=short, **metrics))
    return rows, predictions


def run_scope_comparison(out_dir, *, repeats=3, folds=5, epochs=100, patience=10,
                         seed=42, device="cpu", threads=1, data_dir=None):
    """Run every declared fit and save checkpoints, scalers and predictions.

    Defaults preserve the revised scope notebook's MLP and optimization policy.
    Scope preference is the largest mean restored-checkpoint validation AUROC,
    with shallower scope breaking exact ties. Test scores do not select a scope.
    This is not nested evaluation of an adaptive scope-selection algorithm.
    ``data_dir`` overrides DTP_KG_DATA_DIR; input files are read without changes.
    Record-level outputs and checkpoints belong in a private run directory.
    """
    out = Path(out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new empty run directory: {out}")
    for name, value, minimum in (("repeats", repeats, 1), ("folds", folds, 2),
                                 ("epochs", epochs, 1), ("patience", patience, 1),
                                 ("threads", threads, 1)):
        if int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if repeats > 100 or folds > 1000 or epochs > 10000 or int(seed) != seed or seed < 0:
        raise ValueError("Run coordinates exceed the deterministic sampler seed range")
    data_root = DATA_DIR if data_dir is None else Path(data_dir).expanduser().resolve()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    pos_path = data_root / "interactions/DDI_positive_pairs.csv"
    neg_path = data_root / "interactions/DDI_negative_pairs.csv"
    emb_dir = data_root / "mesh"
    inputs = [pos_path, neg_path, emb_dir / "extended_drug_info.csv"]
    embedding_paths = {s.level_key: emb_dir / f"MeSH_{s.level_key}_tfidf_svd128.csv" for s in SCOPES}
    inputs += list(embedding_paths.values())
    # Hash the package code actually executed, including extracted sampling logic.
    package_root = Path(__file__).resolve().parents[1]
    code = ["mesh_scope/experiment.py", "mesh_scope/protocol.py", "mesh_scope/sampling.py",
            "mesh_scope/training.py", "mesh_scope/dataloader.py", "models/mesh_mlp.py",
            "mesh_scopes.py", "data_loaders.py", "ddi_sampling.py", "ddi_labels.py",
            "evaluation_inputs.py", "evaluation_stats.py", "metrics.py", "project_paths.py"]
    input_hashes = {str(p.relative_to(data_root)): sha256(p) for p in inputs}
    code_hashes = {p: sha256(package_root / p) for p in code}
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(seed=seed, repeats=repeats, folds=folds, epochs=epochs, patience=patience,
        device=device, threads=threads, pair_representation="[a+b, abs(a-b)]",
        positive_sampling="per_epoch", positive_to_negative_ratio=1.0,
        batch_size=128, hidden_dims=[256, 128, 64], batchnorm=True, dropout=.3,
        optimizer="Adam", lr=.001, weight_decay=0.0, loss="BCELoss on sigmoid output",
        validation_fraction=.15, checkpoint_metric="validation AUROC", min_delta=.0001,
        decision_threshold=.5,
        scaler="StandardScaler fitted incrementally to the full training schedule (all epochs' "
               "training pairs; held-out pairs never enter it) and frozen",
        scope_selection="maximum mean restored-checkpoint validation AUROC; exact ties prefer shallower",
        inference="approximate corrected paired CV t; Holm over 6 categories x 3 contrasts x 4 metrics",
        inference_metrics=list(METRICS), inference_family_size=72,
        graph="not used: MeSH-only scope experiment",
        limitations=["Common-coverage balanced benchmark, not deployment prevalence",
                     "Globally prepared fixed MeSH embeddings; transductive representation",
                     "Training reservoirs overlap across folds; shared-drug dependence not fully corrected",
                     "Not nested evaluation of scope selection; selected-scope test results remain exploratory"],
        python=platform.python_version(), torch=torch.__version__,
        input_sha256=input_hashes, input_path_base="data_dir",
        code_sha256=code_hashes, code_path_base="dtpkg")
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    data, common = load_paired_scope_data(LEVELS, pos_path, neg_path, emb_dir, seed=seed, n_splits=folds)
    pairs = data[LEVELS[0]]["pairs"]
    pairs.to_csv(out / "paired_cohort_pairs.csv", index=False)
    pd.DataFrame(sorted(common), columns=["drugbank_id"]).to_csv(out / "common_scope_drugs.csv", index=False)
    cats = load_drug_depths(emb_dir / "extended_drug_info.csv")
    validate_drug_depths(cats, common)
    pool = load_training_positive_pool(pos_path, neg_path, common)
    splits = create_inner_splits(data[LEVELS[0]]["y"], data[LEVELS[0]]["folds"],
        val_frac=.15, seed=seed, num_repetitions=repeats, repeat_outer=True)
    manifest = save_split_manifest(splits, pairs, out / "split_manifest.json", seed=seed, val_frac=.15,
        extra={"shared_across_scopes": True, "positive_sampling": "per_epoch",
               "positive_pool_pairs": len(pool), "positive_pool_hash": frame_hash(pool)})
    assert all(not any(f["overlap_unordered_pairs"].values())
               for rep in manifest["repetitions"] for f in rep)
    embeddings = {level: pd.read_csv(path, index_col=0) for level, path in embedding_paths.items()}
    if len({frame.shape[1] for frame in embeddings.values()}) != 1:
        raise ValueError("All scopes must have equal embedding dimensions for paired initialization")
    sampling = SamplingLoaders(positive_pool=pool, seed=seed, batch_size=128,
                               positive_sampling="per_epoch")
    test_rows, val_rows, logs, epoch_rows, schedule_rows = [], [], [], [], []
    predictions = []
    for rep, fold_splits in enumerate(splits):
        for fold, (tr, va, te) in enumerate(fold_splits):
            train_pairs, val_pairs, test_pairs = [pairs.iloc[idx].reset_index(drop=True) for idx in (tr, va, te)]
            heldout = pd.concat([val_pairs, test_pairs], ignore_index=True)
            sampler = sampling.make_sampler(train_pairs, heldout, embeddings[LEVELS[0]], fold, rep)
            # One sampler object for all scopes guarantees the same eligible pool.
            epoch_hashes = {}
            initialization_hash = None
            for level in LEVELS:
                fit_start = time.monotonic()
                fit_dir = out / "fits" / f"repeat_{rep}_fold_{fold}" / level
                fit_dir.mkdir(parents=True)
                emb = embeddings[level]
                train_loader, val_loader, test_loader, epoch_loader, scaler = sampling.sampled_loaders(
                    sampler, emb, val_pairs, test_pairs, scaler_epochs=epochs)
                # Audit the standardized inputs over every epoch: an epoch-zero scaler gave
                # the Intermediate scope inputs of 1e5 in later epochs (SamplingLoaders.sampled_loaders).
                z_max = {"train": 0.0,
                         "heldout": max(float(l.dataset.X.abs().max()) for l in (val_loader, test_loader))}
                seen = set()
                def audited_loader(epoch):
                    sampled = sampler.epoch(epoch)
                    digest = frame_hash(sampled)
                    if epoch in epoch_hashes:
                        assert epoch_hashes[epoch] == digest, "scope schedules diverged"
                    epoch_hashes[epoch] = digest
                    keys = set(zip(sampled.drug1, sampled.drug2))
                    assert not keys & sampler.heldout
                    positive = sampled[sampled.label == 1]
                    seen.update(zip(positive.drug1, positive.drug2))
                    schedule_rows.append(dict(feature_set=level, split_repeat=rep, fold=fold,
                        epoch=epoch + 1, ordered_pairs_sha256=digest, unique_positives_seen=len(seen)))
                    loader = epoch_loader(epoch)
                    z_max["train"] = max(z_max["train"], float(loader.dataset.X.abs().max()))
                    return loader
                fit_seed = seed + rep * 1000 + fold
                torch.manual_seed(fit_seed)
                np.random.seed(fit_seed)
                model = MLP_DDI(input_dim=data[level]["X"].shape[1], hidden_dims=(256, 128, 64), dropout=.3)
                initial = hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
                if initialization_hash is not None:
                    assert initial == initialization_hash, "scope initializations diverged"
                initialization_hash = initial
                model, history = train_one_fold(model, train_loader, val_loader,
                    n_epochs=epochs, patience=patience, lr=.001, device=device,
                    epoch_loader=audited_loader, verbose=False)
                # Score restored validation and test checkpoints; selection uses validation only.
                val_scores, test_scores = predict(model, val_loader, device), predict(model, test_loader, device)
                vm, vp = score_pairs(val_pairs, val_scores, cats)
                tm, tp = score_pairs(test_pairs, test_scores, cats)
                assert np.isclose(next(r["auc"] for r in vm if r["pair_bin"] == "overall"), history["best_val_auc"])
                reverse = test_pairs.rename(columns={"drug1": "drug2", "drug2": "drug1"})
                forward_x, _, _ = build_pair_features(emb, test_pairs)
                reverse_x, _, _ = build_pair_features(emb, reverse)
                assert np.array_equal(forward_x, reverse_x), "pair-swap invariance failed"
                metadata = dict(feature_set=level, split_repeat=rep, fold=fold, training_seed=fit_seed,
                    inference_protocol="cv", n_outer_train=len(tr) + len(va), n_outer_test=len(te))
                test_rows.extend(dict(**metadata, **r) for r in tm)
                val_rows.extend(dict(**metadata, **r) for r in vm)
                for phase, pred in (("validation", vp), ("test", tp)):
                    predictions.append(pred.assign(phase=phase, **metadata))
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "metadata": metadata, "best_epoch": history["best_epoch"],
                            "best_val_auc": history["best_val_auc"]}, fit_dir / "checkpoint.pt")
                np.savez(fit_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
                (fit_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                epoch_rows.extend(dict(**metadata, epoch=e + 1,
                    **{k: history[k][e] for k in ("train_loss", "val_loss", "val_auc")})
                    for e in range(len(history["val_auc"])))
                log = dict({**metadata, **sampler.describe()}, best_epoch=history["best_epoch"],
                    best_val_auc=history["best_val_auc"], epochs_run=len(history["val_auc"]),
                    hit_epoch_cap=len(history["val_auc"]) == epochs,
                    unique_positives_seen=len(seen), initial_weights_sha256=initial,
                    scaler_min_scale=float(scaler.scale_.min()),
                    standardized_abs_max_train=z_max["train"], standardized_abs_max_heldout=z_max["heldout"],
                    pair_swap_max_feature_difference=0.0, seconds=time.monotonic() - fit_start)
                logs.append(log)
                pd.DataFrame(test_rows).to_csv(out / "results_pair_bins_per_fold.csv", index=False)
                pd.DataFrame(val_rows).to_csv(out / "validation_metrics.csv", index=False)
                pd.DataFrame(logs).to_csv(out / "training_log.csv", index=False)
                pd.DataFrame(epoch_rows).to_csv(out / "epoch_history.csv", index=False)
                pd.DataFrame(schedule_rows).to_csv(out / "sampling_audit.csv", index=False)
                pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
                auc = next(r["auc"] for r in tm if r["pair_bin"] == "overall")
                print(f"FIT {len(logs)}/{repeats * folds * 3}: {NAMES[level]}, repeat {rep}, fold {fold}; "
                      f"selected epoch {history['best_epoch']}/{len(history['val_auc'])}, "
                      f"val AUC {history['best_val_auc']:.4f}, test AUC {auc:.4f}; {log['seconds']:.1f}s", flush=True)
    report_scope_results(out)
    # Fail if inputs or executed source changed while the experiment ran.
    for group, root in (("input_sha256", data_root), ("code_sha256", package_root)):
        if not all(sha256(root / path) == value for path, value in plan[group].items()):
            raise RuntimeError(f"Files in {group} changed during the experiment")
    completion = dict(status="complete", fits=len(logs), seconds=time.monotonic() - start,
        cohort_pairs=len(pairs), cohort_drugs=len(set(pairs.drug1) | set(pairs.drug2)),
        common_embedding_drugs=len(common), eligible_positive_pool=len(pool),
        all_schedule_prefixes_matched=True, all_initial_weights_matched=True,
        all_heldout_exclusions_passed=True, all_pair_swap_checks_passed=True)
    (out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    return pd.DataFrame(test_rows)


def report_scope_results(out_dir):
    """Refresh numerical summaries from saved folds without creating figures.

    The notebook renders the selection figure and exploratory heatmap
    separately, using these saved results without retraining.
    """
    out = Path(out_dir)
    results = normalize_results(pd.read_csv(out / "results_pair_bins_per_fold.csv").rename(columns={"feature_set": "model"}))
    comparisons = comparison_table(results, CONTRASTS, METRICS,
                                  family="scope_exploratory_72", categories=CATEGORIES)
    summary = performance_summary(results, METRICS)
    comparisons.to_csv(out / "scope_comparisons_holm.csv", index=False)
    summary.to_csv(out / "scope_summary_ci.csv", index=False)
    # Overall contrasts are descriptive here; the 72 planned category hypotheses are unchanged.
    logs = pd.read_csv(out / "training_log.csv")
    selection = logs.groupby("feature_set").best_val_auc.agg(["mean", "std", "count"]).reindex(LEVELS)
    selection["scope"] = [NAMES[k] for k in selection.index]
    winner = selection["mean"].idxmax()
    selection["preferred_by_validation_auc"] = selection.index == winner
    selection.to_csv(out / "validation_scope_selection.csv")
    overall = results[results.category.eq("overall")].groupby("model")[list(METRICS)].mean()
    overall.index.name = "feature_set"
    overall["validation_auc"] = selection["mean"]
    overall.to_csv(out / "overall_scope_summary.csv")
    print("Validation scope comparison:\n" + selection.to_string(), flush=True)
