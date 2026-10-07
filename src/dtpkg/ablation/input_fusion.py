"""Raw-feature concatenation versus MeSH and encoded fusion on matched folds.

Extends the completed fusion study without changing its fitting code/artifacts.
New raw concat matches the MeSH baseline's scaling and batch-128 optimizer;
batch-1 controls separate update schedule from representation architecture.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from dtpkg.ablation.artifacts import digest, file_hash, snapshot, validate_completed, write_json
from dtpkg.ablation.reporting import IDENTITY
from dtpkg.ablation.workflow import _metrics
from dtpkg.ddi_labels import POLICY
from dtpkg.ddi_sampling import epoch_seed
from dtpkg.fusion.trainer import FusionTrainer
from dtpkg.models.latent_gate import Classifier, LatentGateModel
from dtpkg.data_loaders import _resolve_supervision_labels, build_pair_features
from dtpkg.ablation.artifacts import PACKAGE_ROOT

REUSED = ("mesh_only", "adaptive_vector", "concat_projection")
NEW = ("raw_concat", "encoded_mesh", "raw_concat_pairwise", "mesh_only_pairwise")
ARMS = (*REUSED, *NEW)
MAIN_CONTRASTS = [("raw_concat", "mesh_only"), ("adaptive_vector", "mesh_only"),
                  ("adaptive_vector", "raw_concat")]
CONTROL_CONTRASTS = [("raw_concat_pairwise", "mesh_only_pairwise"),
                     ("encoded_mesh", "mesh_only_pairwise"),
                     ("adaptive_vector", "raw_concat_pairwise"),
                     ("adaptive_vector", "encoded_mesh"),
                     ("concat_projection", "raw_concat_pairwise"),
                     ("concat_projection", "encoded_mesh"),
                     ("adaptive_vector", "concat_projection")]
CODE = ("ablation/input_fusion.py", "ablation/input_fusion_models.py", "fusion/trainer.py",
        "models/latent_gate.py", "data_loaders.py", "ddi_sampling.py", "ddi_labels.py",
        "metrics.py", "ablation/artifacts.py", "evaluation_stats.py", "ablation/workflow.py", "evaluation_inputs.py")


class ColumnLogitHead(torch.nn.Module):
    """Preserve a column axis for the baseline trainer's squeeze at batch one."""
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, values):
        return self.classifier(values).reshape(-1, 1)


def read_protocol(output_dir):
    p = json.loads((Path(output_dir) / "protocol.json").read_text())
    if p["protocol_id"] != digest({k: v for k, v in p.items() if k != "protocol_id"}):
        raise ValueError("Protocol digest mismatch")
    return p


def _folder(root, arm, rep, fold):
    return Path(root) / "fits" / arm / f"repeat_{rep:02d}_fold_{fold:02d}"


def _identity(p, arm, rep, fold):
    return {key: p[key] for key in IDENTITY} | dict(arm=arm, split_repeat=rep, fold=fold)


def _assert_code(p):
    if any(file_hash(PACKAGE_ROOT / name) != expected for name, expected in p["code_hashes"].items()):
        raise ValueError("Fitting code changed; use a new output directory")


def prepare_input_fusion(source_dir, output_dir):
    """Verify source fits, declare contrasts and save eligible training pool; no fit."""
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from original experiment")
    if (output / "protocol.json").exists():
        p = read_protocol(output)
        _assert_code(p)
        if p["reference_dir"] != str(source):
            raise ValueError("Changed source directory")
        if file_hash(source / "protocol.json") != p["reference_protocol_sha256"]:
            raise ValueError("Changed source protocol")
        if file_hash(output / "positive_pool.csv.gz") != p["positive_pool_sha256"]:
            raise ValueError("Changed eligible positive pool")
        return p
    original = read_protocol(source)
    if original["study"] != "fusion_methods":
        raise ValueError("Require the revised fusion-method source experiment")
    for name in ("fusion/trainer.py", "models/latent_gate.py", "data_loaders.py", "ddi_sampling.py", "metrics.py"):
        if file_hash(PACKAGE_ROOT / name) != original["code_hashes"][name]:
            raise ValueError(f"Current training implementation differs from source: {name}")
    if file_hash(source / "split_manifest.json") != original["split_manifest_hash"]:
        raise ValueError("Source split manifest changed")
    coordinates = sorted({(x["split_repeat"], x["fold"]) for x in original["expected_fits"]})
    if coordinates != [(r, f) for r in range(5) for f in range(5)]:
        raise ValueError("Require complete five-repeat/five-fold source")
    source_fits = []
    for rep, fold in coordinates:
        for arm in REUSED:
            folder = _folder(source, arm, rep, fold)
            if not validate_completed(folder, _identity(original, arm, rep, fold)):
                raise ValueError(f"Incomplete reference: {folder}")
            source_fits.append(dict(arm=arm, split_repeat=rep, fold=fold,
                                    completion_sha256=file_hash(folder / "completed.json")))
    for name, record in original["source_files"].items():
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError(f"Original scientific input changed: {name}")
    positive, _ = _resolve_supervision_labels(pd.read_csv(original["source_files"]["positives"]["path"]),
        pd.read_csv(original["source_files"]["negatives"]["path"]), POLICY, None)
    embeddings = pd.read_csv(original["source_files"]["embeddings"]["path"], index_col=0)
    ids = set(embeddings.index.astype(str))
    pool = positive[positive.drug1.isin(ids) & positive.drug2.isin(ids)].copy().assign(label=1)
    output.mkdir(parents=True, exist_ok=True)
    pool.to_csv(output / "positive_pool.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    shutil.copyfile(source / "split_manifest.json", output / "split_manifest.json")
    config = dict(original["scientific_config"], study="input_fusion")
    arms = []
    for arm in ARMS:
        batch = 128 if arm in ("mesh_only", "raw_concat") else 1
        arms.append(dict(name=arm, model_kind="mesh" if arm == "mesh_only" else "fusion",
                         optimizer_batch_size=batch, reference_only=False,
                         source="verified_reference" if arm in REUSED else "new_fit"))
    p = dict(format_version=1, study="input_fusion", scientific_config=config,
             data_identity=original["data_identity"], split_manifest_hash=original["split_manifest_hash"],
             source_files=original["source_files"], reference_dir=str(source),
             reference_protocol_sha256=file_hash(source / "protocol.json"), reference_fits=source_fits,
             positive_pool_sha256=file_hash(output / "positive_pool.csv.gz"),
             code_hashes={name: file_hash(PACKAGE_ROOT / name) for name in CODE},
             arms=arms, contrasts=MAIN_CONTRASTS, control_contrasts=CONTROL_CONTRASTS,
             primary_metrics=["auc", "f1"], category_metrics=["auc", "f1"], confidence=.95,
             coordinates=coordinates,
             expected_fits=[dict(arm=arm, split_repeat=r, fold=f) for r, f in coordinates for arm in ARMS],
             raw_concat="No learned modality encoders; concatenate input features per drug, then sum/absolute difference and MLP",
             raw_scaling="StandardScaler of symmetric pair features fitted on epoch-zero inner-training pairs only; frozen",
             optimization="Raw concat batch128 matches MeSH baseline; matched batch1 controls use same sampled pairs and pair scaling",
             encoded_mesh="Same raw MeSH encoder and symmetric classifier as fusion; no topology input used",
             initialization="Raw heads share initialization across update regimes; encoded MeSH encoder/head copied from seeded fusion initialization",
             checkpoint_policy="100 epochs maximum, patience10,min_delta1e-4,inner-validation BCE; no test-based tuning",
             graph_policy=original["graph_policy"],
             inference_families={"primary_overall": 6, "secondary_controls": 14, "exploratory_depth": 36},
             interpretation="Follow-up on known cohort; system-level comparisons do not establish MeSH dominance or causal modality information",
             environment=original["environment"])
    p["protocol_id"] = digest(p)
    write_json(output / "protocol.json", p)
    snapshot(PACKAGE_ROOT, output / "code_snapshot.zip", p["code_hashes"])
    print(f"Prepared {len(pool)} eligible positives;75 reference fits,100 new fits", flush=True)
    return p


def _validate_reference(p, arm, rep, fold):
    folder = _folder(p["reference_dir"], arm, rep, fold)
    record = next(x for x in p["reference_fits"] if (x["arm"], x["split_repeat"], x["fold"]) == (arm, rep, fold))
    if file_hash(folder / "completed.json") != record["completion_sha256"]:
        raise ValueError("Reference fit completion record changed")
    original = read_protocol(p["reference_dir"])
    if not validate_completed(folder, _identity(original, arm, rep, fold)):
        raise ValueError("Incomplete reference fit")
    return folder


def _seal(folder, identity):
    files = [f for f in folder.iterdir() if f.is_file() and f.name not in ("completed.json", ".running")
             and not f.name.endswith(".tmp")]
    write_json(folder / "completed.json", dict(**identity, artifact_hashes={f.name: file_hash(f) for f in files}))


def _reuse(p, output, arm, rep, fold):
    source = _validate_reference(p, arm, rep, fold)
    folder = _folder(output, arm, rep, fold)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("fold_metrics.csv", "predictions.csv", "validation_predictions.csv", "history.csv", "gates.csv", "sampling.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, folder / name)
    info = json.loads((source / "fit_info.json").read_text())
    info.update(source="verified_reference", reference_path=str(source),
                optimizer_batch_size=128 if arm == "mesh_only" else 1)
    write_json(folder / "fit_info.json", info)
    write_json(folder / "reference.json", dict(path=str(source), completed_sha256=file_hash(source / "completed.json"),
                                               checkpoint_sha256=file_hash(source / "checkpoint.pt")))
    _seal(folder, _identity(p, arm, rep, fold))


def _encoded_factory(config, bio_dim, topo_dim):
    from dtpkg.ablation.input_fusion_models import EncodedMeshModel
    kwargs = dict(bio_dim=bio_dim, topo_dim=topo_dim, latent_dim=config["latent_dim"],
                  enc_hidden=tuple(config["enc_hidden"]), head_hidden=tuple(config["head_hidden"]),
                  dropout=config["dropout"], fusion_mode="sym", gate_bias_init=.7)
    def factory():
        reference = LatentGateModel(**kwargs)
        after = torch.get_rng_state()
        model = EncodedMeshModel(bio_dim=bio_dim, latent_dim=config["latent_dim"],
                                 enc_hidden=tuple(config["enc_hidden"]), head_hidden=tuple(config["head_hidden"]),
                                 dropout=config["dropout"])
        model.bio_encoder.load_state_dict(reference.bio_encoder.state_dict())
        model.classifier.load_state_dict(reference.classifier.state_dict())
        torch.set_rng_state(after)
        return model
    return factory


def _raw_scores(head, scaler, features, pairs):
    x, _, kept = build_pair_features(features, pairs)
    if len(kept) != len(pairs):
        raise ValueError("Missing feature coverage")
    x = torch.as_tensor(scaler.transform(x), dtype=torch.float32)
    head.eval()
    with torch.inference_mode():
        return torch.sigmoid(head(x)).cpu().numpy().reshape(-1)


def _encoded_scores(model, embeddings, topology, pairs):
    from dtpkg.ablation.workflow import _predict
    return _predict(model, "fusion", pairs, embeddings, topology, "cpu")


def _fit(p, output, arm, rep, fold, pool, bundle, template_folder):
    from dtpkg.ablation.input_fusion_models import RawConcatModel
    c = p["scientific_config"]
    folder = _folder(output, arm, rep, fold)
    folder.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle[k] for k in ("train_pairs", "validation_pairs", "test_pairs"))
    embeddings, topology = bundle["embeddings"], bundle["topology"]
    batch = 128 if arm == "raw_concat" else 1
    topo_dim = 0 if arm == "mesh_only_pairwise" else topology.shape[1]
    trainer = FusionTrainer(device="cpu", epochs=c["epochs"], patience=c["patience"], min_delta=c["min_delta"],
        lr=c["lr"], weight_decay=c["weight_decay"], batch_size=batch, seed=c["seed"],
        val_frac=.15, positive_sampling="per_epoch", positive_to_negative_ratio=1.0,
        positive_pool=pool, out_dir=str(folder), drop_bio_prob=0., drop_topo_prob=0.)
    sampler = trainer.make_sampler(train, pd.concat([val, test]), embeddings, fold, rep)
    if sampler.describe() != bundle["sampling"]:
        raise ValueError("New fit sampling differs from reference schedule")
    fit_seed = epoch_seed(c["seed"], fold, rep, 0)
    started = time.monotonic()
    scaler = None
    if arm == "encoded_mesh":
        _, model = trainer._train_fusion(_encoded_factory(c, embeddings.shape[1], topology.shape[1]),
            train, val, topology, embeddings, bundle["depths"], epoch_sampler=sampler, fit_seed=fit_seed)
        validation_scores = _encoded_scores(model, embeddings, topology, val)
        scores = _encoded_scores(model, embeddings, topology, test)
        swap_scores = _encoded_scores(model, embeddings, topology, test.rename(columns={"drug1": "drug2", "drug2": "drug1"}))
    else:
        if topo_dim:
            features = embeddings.join(topology, how="left", rsuffix="_topology")
        else:
            features = embeddings
        def factory():
            return ColumnLogitHead(RawConcatModel(bio_dim=embeddings.shape[1], topo_dim=topo_dim,
                                  head_hidden=tuple(c["head_hidden"]), dropout=c["dropout"]).classifier)
        trainer.baseline_cls = factory
        tl, vl, _, el, scaler = trainer.sampled_loaders(sampler, features, val)
        _, head = trainer._train_baseline(tl, vl, epoch_sampler=sampler, epoch_loader=el, fit_seed=fit_seed)
        validation_scores, scores = _raw_scores(head, scaler, features, val), _raw_scores(head, scaler, features, test)
        swap_scores = _raw_scores(head, scaler, features, test.rename(columns={"drug1": "drug2", "drug2": "drug1"}))
        model = RawConcatModel(bio_dim=embeddings.shape[1], topo_dim=topo_dim,
                               head_hidden=tuple(c["head_hidden"]), dropout=c["dropout"],
                               pair_mean=scaler.mean_, pair_scale=scaler.scale_)
        model.classifier.load_state_dict(head.classifier.state_dict())
    if not np.isfinite(scores).all() or not np.isfinite(validation_scores).all():
        raise ValueError("Nonfinite model predictions")
    swap_error = float(np.max(np.abs(scores-swap_scores)))
    if swap_error > 1e-7:
        raise ValueError("Pair order changes predictions")
    meta = dict(model=arm, split_repeat=rep, fold=fold, training_seed=fit_seed,
                inference_protocol="cv", n_outer_train=len(train)+len(val), n_outer_test=len(test))
    outputs = []
    for name, frame, values in (("validation_predictions.csv", val, validation_scores), ("predictions.csv", test, scores)):
        original = pd.read_csv(template_folder / name)
        pd.testing.assert_frame_equal(original[["drug1", "drug2", "label"]], frame[["drug1", "drug2", "label"]].reset_index(drop=True), check_dtype=False)
        new = original.assign(**meta, score=values)
        new.to_csv(folder / name, index=False)
        outputs.append(new)
    _metrics(outputs[-1]).assign(**meta).to_csv(folder / "fold_metrics.csv", index=False)
    info = dict(trainer.last_fit_info, **meta, source="fitted", optimizer_batch_size=batch,
        parameter_count=sum(x.numel() for x in model.parameters()), max_swap_score_difference=swap_error,
        unique_positive_exposure_selected_checkpoint=sampler.unique_positives(trainer.last_fit_info["best_epoch"]),
        unique_positive_exposure_stopped_epoch=sampler.unique_positives(trainer.last_fit_info["epochs_run"]),
        wall_seconds=time.monotonic()-started, topology_scaling="none_for_encoder" if arm == "encoded_mesh" else "pair_standardized",
        normalization_population="none" if scaler is None else "epoch_zero_inner_training_pairs")
    write_json(folder / "fit_info.json", info)
    write_json(folder / "sampling.json", sampler.describe())
    pd.DataFrame(trainer.last_epoch_losses).to_csv(folder / "history.csv", index=False)
    checkpoint = dict(identity=_identity(p, arm, rep, fold), model=model.cpu().eval(), pair_scaler=scaler,
                      feature_columns=list(embeddings.columns) + (list(topology.columns) if topo_dim else []),
                      prediction_path="encoded_forward" if arm == "encoded_mesh" else "pair_scaler_then_classifier",
                      source_checkpoint=str(template_folder / "checkpoint.pt"),
                      source_checkpoint_sha256=file_hash(template_folder / "checkpoint.pt"), fit_info=info)
    torch.save(checkpoint, folder / "checkpoint.pt")
    _assert_code(p)
    _seal(folder, _identity(p, arm, rep, fold))
    print(f"Completed {arm}:repeat{rep},fold{fold},epochs{info['epochs_run']},seconds{info['wall_seconds']:.1f}", flush=True)


def run_input_fusion(output_dir, coordinate=None):
    """Run all folds, or one array coordinate 0..24, using frozen protocol choices."""
    output = Path(output_dir)
    p = read_protocol(output)
    _assert_code(p)
    if file_hash(output / "split_manifest.json") != p["split_manifest_hash"]:
        raise ValueError("Saved split manifest changed")
    if file_hash(output / "positive_pool.csv.gz") != p["positive_pool_sha256"]:
        raise ValueError("Saved positive pool changed")
    torch.set_num_threads(1)
    if coordinate is not None and not 0 <= int(coordinate) < 25:
        raise ValueError("coordinate must be in0..24")
    coordinates = p["coordinates"] if coordinate is None else [p["coordinates"][int(coordinate)]]
    pool = pd.read_csv(output / "positive_pool.csv.gz")
    for rep, fold in coordinates:
        lock = output / f".running_{rep}_{fold}"
        with lock.open("x") as f:
            f.write(f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID','local')}\n")
        try:
            template = _validate_reference(p, "adaptive_vector", rep, fold)
            bundle = torch.load(template / "checkpoint.pt", map_location="cpu", weights_only=False)
            for arm in ARMS:
                if validate_completed(_folder(output, arm, rep, fold), _identity(p, arm, rep, fold)):
                    print(f"Reuse completed {arm}:repeat{rep},fold{fold}", flush=True)
                    continue
                print(f"START {arm}:repeat{rep},fold{fold}", flush=True)
                if arm in REUSED:
                    _reuse(p, output, arm, rep, fold)
                else:
                    _fit(p, output, arm, rep, fold, pool, bundle, template)
        finally:
            lock.unlink(missing_ok=True)
    done = sum((_folder(output, x["arm"], x["split_repeat"], x["fold"]) / "completed.json").is_file()
               for x in p["expected_fits"])
    if done == len(p["expected_fits"]):
        write_json(output / "completion.json", dict(**{k: p[k] for k in IDENTITY}, completed_fits=done))
    return dict(completed_fits=done, expected_fits=len(p["expected_fits"]))
