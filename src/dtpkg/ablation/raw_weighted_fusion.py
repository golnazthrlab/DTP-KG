"""Matched-fold extension for fixed and learned raw-input topology weights.

The original α=4 and learned-scalar ideas are fitted on the revised cohort.
Both apply α after the same training-only pair scaler as raw input concat;
otherwise fitting a new scaler would algebraically erase a fixed α.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from dtpkg.ablation.artifacts import digest, file_hash, snapshot, validate_completed, write_json
from dtpkg.ablation.input_fusion import (_folder, _identity, _seal,
                                   _validate_reference, read_protocol)
from dtpkg.ablation.reporting import IDENTITY, _load
from dtpkg.ablation.workflow import _metrics
from dtpkg.ablation.raw_weighted_models import WeightedPairHead
from dtpkg.ddi_sampling import epoch_seed
from dtpkg.fusion.trainer import FusionTrainer
from dtpkg.data_loaders import build_pair_features
from dtpkg.ablation.artifacts import PACKAGE_ROOT

ARMS = ("raw_fixed4_pairwise", "raw_learned_pairwise")
CODE = ("ablation/raw_weighted_fusion.py", "ablation/raw_weighted_models.py",
        "ablation/input_fusion.py", "ablation/input_fusion_models.py",
        "fusion/trainer.py", "models/latent_gate.py", "data_loaders.py",
        "ddi_sampling.py", "ddi_labels.py", "metrics.py", "ablation/artifacts.py",
        "evaluation_stats.py", "ablation/workflow.py", "evaluation_inputs.py")


def _assert_code(protocol):
    changed = [name for name, expected in protocol["code_hashes"].items()
               if file_hash(PACKAGE_ROOT / name) != expected]
    if changed:
        raise ValueError(f"Frozen fitting code changed: {changed}")


def _source_folder(protocol, arm, rep, fold):
    folder = _folder(protocol["source_input_dir"], arm, rep, fold)
    source = read_protocol(protocol["source_input_dir"])
    record = next(x for x in protocol["source_fits"]
                  if (x["arm"], x["split_repeat"], x["fold"]) == (arm, rep, fold))
    if file_hash(folder / "completed.json") != record["completion_sha256"]:
        raise ValueError(f"Source completion record changed: {folder}")
    if not validate_completed(folder, _identity(source, arm, rep, fold)):
        raise ValueError(f"Invalid source fit: {folder}")
    return folder


def prepare_raw_weighted(source_dir, output_dir):
    """Freeze the revised source and comparison plan; no fitting occurs."""
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate")
    if (output / "protocol.json").exists():
        protocol = read_protocol(output)
        _assert_code(protocol)
        if (protocol["source_input_dir"] != str(source)
                or file_hash(source / "protocol.json") != protocol["source_protocol_sha256"]
                or file_hash(output / "positive_pool.csv.gz") != protocol["positive_pool_sha256"]):
            raise ValueError("Prepared source or eligible-positive pool changed")
        return protocol

    source_protocol, fits, missing = _load(source, allow_partial=False)
    if source_protocol["study"] != "input_fusion" or missing or len(fits) != 175:
        raise ValueError("Require all 175 verified input-fusion fits")
    _assert_code(source_protocol)
    coordinates = source_protocol["coordinates"]
    required = ("adaptive_vector", "raw_concat_pairwise")
    source_fits = []
    for rep, fold in coordinates:
        for arm in required:
            folder = _folder(source, arm, rep, fold)
            if not validate_completed(folder, _identity(source_protocol, arm, rep, fold)):
                raise ValueError(f"Incomplete source fit: {folder}")
            source_fits.append(dict(arm=arm, split_repeat=rep, fold=fold,
                                    completion_sha256=file_hash(folder / "completed.json")))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "split_manifest.json", output / "split_manifest.json")
    shutil.copyfile(source / "positive_pool.csv.gz", output / "positive_pool.csv.gz")
    config = dict(source_protocol["scientific_config"], study="raw_weighted_fusion")
    protocol = dict(format_version=1, study="raw_weighted_fusion",
        scientific_config=config, data_identity=source_protocol["data_identity"],
        split_manifest_hash=source_protocol["split_manifest_hash"],
        source_files=source_protocol["source_files"],
        source_input_dir=str(source),
        source_protocol_sha256=file_hash(source / "protocol.json"),
        source_fits=source_fits,
        positive_pool_sha256=file_hash(output / "positive_pool.csv.gz"),
        code_hashes={name: file_hash(PACKAGE_ROOT / name) for name in CODE},
        arms=[dict(name=arm, model_kind="fusion", optimizer_batch_size=1,
                   source="new_fit") for arm in ARMS],
        coordinates=coordinates,
        expected_fits=[dict(arm=arm, split_repeat=rep, fold=fold)
                       for rep, fold in coordinates for arm in ARMS],
        fixed_alpha=4.0, learned_alpha_init=2.0,
        learned_alpha_lr_multiplier=20.0,
        alpha_position="topology coordinates after source raw-concat pair StandardScaler",
        alpha_meaning="one shared scalar across all drugs/features, not a drug-specific gate",
        optimizer="Adam, classifier lr 0.001 and learned alpha lr 0.02 with no alpha weight decay",
        checkpoint_policy="Source sampled pairs and inner validation, max 100 epochs, patience 10, BCE checkpoint selection",
        comparison_families={
            "topology_contribution": [
                ["raw_concat_pairwise", "mesh_only_pairwise"],
                ["raw_fixed4_pairwise", "mesh_only_pairwise"],
                ["raw_learned_pairwise", "mesh_only_pairwise"]],
            "latent_vs_raw": [["adaptive_vector", arm] for arm in
                               ("raw_concat_pairwise", *ARMS)],
            "raw_weighting": [[arm, "raw_concat_pairwise"] for arm in ARMS]},
        comparison_metrics=["auc", "f1"], confidence=.95,
        interpretation="A scale on standardized topology can be absorbed into a first MLP layer; compare optimized pipelines, not representational capacity or missing-data rescue")
    protocol["protocol_id"] = digest(protocol)
    write_json(output / "protocol.json", protocol)
    snapshot(PACKAGE_ROOT, output / "code_snapshot.zip", protocol["code_hashes"])
    print(f"Prepared {len(coordinates)} folds × {len(ARMS)} weighted fits", flush=True)
    return protocol


def _train_weighted(trainer, factory, train_loader, val_loader, epoch_loader,
                    sampler, fit_seed, learned):
    """Mirror the baseline loss/stopping loop, with α's declared optimizer group."""
    model = trainer._fresh_model(factory, fit_seed)
    stopper = trainer._make_stopper()
    if learned:
        optimizer = torch.optim.Adam([
            dict(params=list(model.classifier.parameters()), lr=trainer.lr,
                 weight_decay=trainer.weight_decay),
            dict(params=[model.alpha], lr=trainer.lr * 20.0, weight_decay=0.0)])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer.lr,
                                     weight_decay=trainer.weight_decay)
    trainer.last_epoch_losses = []
    val_losses, best_state = [], None
    for epoch in range(trainer.epochs):
        model.train()
        train_by_class = {0: [], 1: []}
        for xb, yb in epoch_loader(epoch):
            xb, yb = xb.to(trainer.device), yb.to(trainer.device)
            optimizer.zero_grad()
            per = nn.functional.binary_cross_entropy_with_logits(
                model(xb).squeeze(-1), yb, reduction="none")
            for label in (0, 1):
                train_by_class[label].extend(per[yb == label].detach().cpu().tolist())
            per.mean().backward()
            optimizer.step()
        model.eval()
        val_by_class = {0: [], 1: []}
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(trainer.device), yb.to(trainer.device)
                per = nn.functional.binary_cross_entropy_with_logits(
                    model(xb).squeeze(-1), yb, reduction="none")
                losses.extend(per.cpu().tolist())
                for label in (0, 1):
                    val_by_class[label].extend(per[yb == label].cpu().tolist())
        trainer.last_epoch_losses.append(trainer._class_loss_row(
            epoch, train_by_class, val_by_class))
        mean_loss = float(np.mean(losses))
        val_losses.append(mean_loss)
        if stopper.step(mean_loss):
            best_state = deepcopy(model.state_dict())
        if stopper.should_stop:
            break
    if best_state is None:
        raise ValueError("Weighted model did not select a validation checkpoint")
    model.load_state_dict(best_state)
    trainer.last_fit_info = dict(epochs_run=len(val_losses),
        best_epoch=stopper.best_epoch + 1, best_val_loss=float(stopper.best),
        n_train=len(train_loader.dataset), n_val=len(val_loader.dataset),
        n_test=None, positive_sampling=sampler.mode, fit_seed=fit_seed)
    trainer.last_fit_info.update(sampler.describe(len(val_losses)))
    trainer.last_fit_info["unique_training_positives"] = sampler.unique_positives(len(val_losses))
    return model


def _scores(model, scaler, features, pairs):
    values, _, kept = build_pair_features(features, pairs)
    if len(kept) != len(pairs):
        raise ValueError("Missing feature coverage")
    tensor = torch.as_tensor(scaler.transform(values), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1)


def _fit(protocol, output, arm, rep, fold, pool, bundle, raw_source):
    config = protocol["scientific_config"]
    folder = _folder(output, arm, rep, fold)
    folder.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle[key] for key in
                        ("train_pairs", "validation_pairs", "test_pairs"))
    embeddings, topology = bundle["embeddings"], bundle["topology"]
    features = embeddings.join(topology, how="left", rsuffix="_topology")
    trainer = FusionTrainer(device="cpu", epochs=config["epochs"],
        patience=config["patience"], min_delta=config["min_delta"],
        lr=config["lr"], weight_decay=config["weight_decay"], batch_size=1,
        seed=config["seed"], val_frac=.15, positive_sampling="per_epoch",
        positive_to_negative_ratio=1.0, positive_pool=pool, out_dir=str(folder),
        drop_bio_prob=0.0, drop_topo_prob=0.0)
    sampler = trainer.make_sampler(train, pd.concat([val, test]), embeddings, fold, rep)
    if sampler.describe() != bundle["sampling"]:
        raise ValueError("Sampling differs from saved source schedule")
    train_loader, val_loader, _, epoch_loader, scaler = trainer.sampled_loaders(
        sampler, features, val)
    raw_checkpoint = torch.load(raw_source / "checkpoint.pt", map_location="cpu",
                                weights_only=False)
    raw_scaler = raw_checkpoint["pair_scaler"]
    if raw_checkpoint["prediction_path"] != "pair_scaler_then_classifier":
        raise ValueError("Unexpected source raw-concat prediction path")
    for name in ("mean_", "scale_", "var_"):
        if not np.array_equal(getattr(scaler, name), getattr(raw_scaler, name)):
            raise ValueError(f"Weighted fit uses a different unweighted pair scaler: {name}")
    learned = arm == "raw_learned_pairwise"
    alpha_init = protocol["learned_alpha_init"] if learned else protocol["fixed_alpha"]
    def factory():
        return WeightedPairHead(bio_dim=embeddings.shape[1], topo_dim=topology.shape[1],
            alpha_init=alpha_init, learned=learned,
            hidden=tuple(config["head_hidden"]), dropout=config["dropout"])
    fit_seed = epoch_seed(config["seed"], fold, rep, 0)
    started = time.monotonic()
    model = _train_weighted(trainer, factory, train_loader, val_loader, epoch_loader,
                            sampler, fit_seed, learned)
    validation_scores = _scores(model, scaler, features, val)
    test_scores = _scores(model, scaler, features, test)
    swapped = _scores(model, scaler, features,
                      test.rename(columns={"drug1": "drug2", "drug2": "drug1"}))
    if not (np.isfinite(validation_scores).all() and np.isfinite(test_scores).all()):
        raise ValueError("Nonfinite weighted predictions")
    swap_error = float(np.max(np.abs(test_scores - swapped)))
    if swap_error > 1e-7:
        raise ValueError("Pair-order-sensitive weighted predictions")
    metadata = dict(model=arm, split_repeat=rep, fold=fold, training_seed=fit_seed,
        inference_protocol="cv", n_outer_train=len(train)+len(val),
        n_outer_test=len(test))
    for filename, frame, scores in (("validation_predictions.csv", val, validation_scores),
                                    ("predictions.csv", test, test_scores)):
        template = pd.read_csv(raw_source / filename)
        pd.testing.assert_frame_equal(template[["drug1", "drug2", "label"]],
            frame[["drug1", "drug2", "label"]].reset_index(drop=True),
            check_dtype=False)
        template.assign(**metadata, score=scores).to_csv(folder / filename, index=False)
    _metrics(pd.read_csv(folder / "predictions.csv")).assign(**metadata).to_csv(
        folder / "fold_metrics.csv", index=False)
    info = dict(trainer.last_fit_info, **metadata, source="new_fit",
        optimizer_batch_size=1,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        alpha_initial=float(alpha_init), alpha_selected=float(model.alpha_value),
        alpha_learned=learned, alpha_lr_multiplier=20.0 if learned else None,
        max_swap_score_difference=swap_error,
        unique_positive_exposure_selected_checkpoint=sampler.unique_positives(
            trainer.last_fit_info["best_epoch"]),
        unique_positive_exposure_stopped_epoch=sampler.unique_positives(
            trainer.last_fit_info["epochs_run"]),
        normalization_population="epoch_zero_inner_training_pairs",
        wall_seconds=time.monotonic()-started)
    write_json(folder / "fit_info.json", info)
    write_json(folder / "sampling.json", sampler.describe())
    pd.DataFrame(trainer.last_epoch_losses).to_csv(folder / "history.csv", index=False)
    torch.save(dict(identity=_identity(protocol, arm, rep, fold), model=model.cpu().eval(),
        pair_scaler=scaler, feature_columns=list(features.columns),
        prediction_path="unweighted_pair_scaler_then_weighted_classifier",
        source_raw_checkpoint=str(raw_source / "checkpoint.pt"),
        source_raw_checkpoint_sha256=file_hash(raw_source / "checkpoint.pt"),
        fit_info=info), folder / "checkpoint.pt")
    _assert_code(protocol)
    _seal(folder, _identity(protocol, arm, rep, fold))
    print(f"Completed {arm}:repeat{rep},fold{fold},alpha={model.alpha_value:.5g},"
          f"epochs={info['epochs_run']},seconds={info['wall_seconds']:.1f}", flush=True)


def run_raw_weighted(output_dir, coordinate=None):
    output = Path(output_dir)
    protocol = read_protocol(output)
    _assert_code(protocol)
    if (file_hash(output / "split_manifest.json") != protocol["split_manifest_hash"]
            or file_hash(output / "positive_pool.csv.gz") != protocol["positive_pool_sha256"]):
        raise ValueError("Saved split manifest or positive pool changed")
    if coordinate is not None and not 0 <= int(coordinate) < 25:
        raise ValueError("coordinate must be 0..24")
    torch.set_num_threads(1)
    coordinates = (protocol["coordinates"] if coordinate is None else
                   [protocol["coordinates"][int(coordinate)]])
    pool = pd.read_csv(output / "positive_pool.csv.gz")
    for rep, fold in coordinates:
        lock = output / f".running_{rep}_{fold}"
        with lock.open("x") as stream:
            stream.write(f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID','local')}\n")
        try:
            template = _source_folder(protocol, "adaptive_vector", rep, fold)
            raw = _source_folder(protocol, "raw_concat_pairwise", rep, fold)
            source_protocol = read_protocol(protocol["source_input_dir"])
            original = _validate_reference(source_protocol, "adaptive_vector", rep, fold)
            bundle = torch.load(original / "checkpoint.pt", map_location="cpu",
                                weights_only=False)
            for arm in ARMS:
                folder = _folder(output, arm, rep, fold)
                if validate_completed(folder, _identity(protocol, arm, rep, fold)):
                    continue
                print(f"START {arm}:repeat{rep},fold{fold}", flush=True)
                _fit(protocol, output, arm, rep, fold, pool, bundle, raw)
        finally:
            lock.unlink(missing_ok=True)
    done = sum((_folder(output, record["arm"], record["split_repeat"],
                         record["fold"]) / "completed.json").is_file()
               for record in protocol["expected_fits"])
    if done == len(protocol["expected_fits"]):
        write_json(output / "completion.json",
                   dict(**{key: protocol[key] for key in IDENTITY}, completed_fits=done))
    return dict(completed_fits=done, expected_fits=len(protocol["expected_fits"]))
