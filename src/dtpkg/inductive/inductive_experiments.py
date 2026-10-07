"""Fit and safely resume the published repeated drug-holdout experiment."""
from dataclasses import asdict
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import torch

from dtpkg.evaluation_inputs import load_drug_depths
from dtpkg.inductive.graph_access import build_split_graphs
from dtpkg.inductive.inductive_data import file_hash, json_hash, write_json
from dtpkg.inductive.inductive_fit import CODE_FILES, DEFAULT_SETTINGS, fit_split_seed
from dtpkg.inductive.inductive_inference import recover_ancillary_exports, score_saved_fit
from dtpkg.inductive.validation import validate_experiment, load_prepared_split
from dtpkg.mesh_scopes import SCOPE_BY_KEY
from dtpkg.project_paths import DATA_DIR
from dtpkg.topology_config import CANONICAL_GRAPH, DENSE_GRAPH_SETTINGS, make_extractor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARM = dict(name="biological_only", training_graph_variant="no_ddi", heldout_ddi_policy="none",
           role="primary", context_fraction=None, context_seed=None)


def validate_output_location(prepared_dir, results_dir, data_dir=None, graph_path=None):
    """Keep run artifacts outside immutable preparation and source-data locations."""
    prepared, out = Path(prepared_dir).resolve(), Path(results_dir).resolve()
    data = Path(DATA_DIR if data_dir is None else data_dir).expanduser().resolve()
    if out == prepared or out in prepared.parents or prepared in out.parents:
        raise ValueError("Results and prepared split directories must not overlap")
    if out == data or out in data.parents:
        raise ValueError("Results cannot replace the source data directory")
    protected = [data / name for name in ("mesh", "interactions", "networks", "drugbank")]
    if graph_path is not None:
        protected.append(Path(graph_path).expanduser().resolve())
    for source in protected:
        source = source.resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Results must not overlap source input locations")
    return out


def claim_output_directory(path, identity):
    """Recognize a resumable package output before writing any run artifacts."""
    path = Path(path)
    marker = path / "inductive_run.json"
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not marker.is_file():
            raise FileExistsError(f"Unrecognized nonempty results directory: {path}; use a new directory")
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"Saved fit settings differ: {path}; use a new results directory")
    else:
        path.mkdir(parents=True, exist_ok=True)
        write_json(marker, identity)


def attach_metadata(predictions, manifest, training_seed, fit_identity):
    """One fit scores both endpoint settings; each result retains its split identity."""
    p = predictions.copy()
    heldout = set(manifest["heldout_drugs"])
    nheld = p.drug1.isin(heldout).astype(int) + p.drug2.isin(heldout).astype(int)
    if not nheld.isin([1, 2]).all():
        raise ValueError("inductive pair without a held-out endpoint")
    p["scenario"] = np.where(nheld == 1, "seen_unseen", "unseen_unseen")
    p["split_id"] = manifest["split_id"]
    p["split_repeat"] = manifest["split_repeat"]
    p["training_seed"] = training_seed
    p["n_heldout_drugs"] = manifest["n_heldout"]
    p["n_development_drugs"] = manifest["n_development"]
    p["inference_protocol"] = "inductive_repeated_drug_holdout"
    p["analysis_arm"] = p["source_arm"] = "biological_only"
    p["fit_identity"] = fit_identity
    return p


def run_prepared_splits(prepared_dir, results_dir, split_ids=None, training_seeds=(101, 202, 303),
                        settings=None, graph_path=None, *, data_dir=None, n_jobs=16,
                        device=None, fit_missing=True, threads=None):
    """Fit three neural methods and the shared-target comparator per split/seed.

    Defaults reproduce the training protocol: 100 epochs, early stopping at ten
    unimproved validation-BCE epochs, epoch-wise balanced positive sampling,
    and transforms fitted over the planned development-only training schedule.
    Both endpoint settings share each fitted model. All outputs are private.

    Existing completed fits can be reused only when source/input/settings hashes
    match. Stale fits fail with a request for a new results directory. Passing
    ``fit_missing=False`` permits reuse and checkpoint recovery without training.
    """
    prepared_dir, results_dir = Path(prepared_dir), Path(results_dir)
    data = Path(DATA_DIR if data_dir is None else data_dir).expanduser().resolve()
    if graph_path is None:
        if os.environ.get("DTP_KG_NETWORK_DIR"):
            graph_path = Path(os.environ["DTP_KG_NETWORK_DIR"]).expanduser() / CANONICAL_GRAPH.name
        else:
            graph_path = CANONICAL_GRAPH if data_dir is None else data / "networks" / CANONICAL_GRAPH.name
    graph_path = Path(graph_path).expanduser().resolve()
    results_dir = validate_output_location(prepared_dir, results_dir, data, graph_path)
    training_seeds = list(training_seeds)
    if not training_seeds or len(set(training_seeds)) != len(training_seeds) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in training_seeds):
        raise ValueError("distinct integer training seeds in [0, 2**32) required")
    for name, value in (("n_jobs", n_jobs), ("threads", threads)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    options = dict(DEFAULT_SETTINGS, **(settings or {}))
    if set(options) - set(DEFAULT_SETTINGS):
        raise ValueError(f"unknown training settings: {set(options) - set(DEFAULT_SETTINGS)}")
    if options['scaling_policy'] != 'planned_training_schedule':
        raise ValueError("published inductive protocol requires planned_training_schedule scaling")
    for name in ('epochs', 'patience', 'batch_size'):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < options['val_split'] < 1:
        raise ValueError("val_split must be between zero and one")
    config = validate_experiment(prepared_dir, graph_path, data_dir=data)
    selection = config.get("selection_protocol", {})
    if not selection or not set(training_seeds).issubset(selection.get("training_seeds", [])):
        raise ValueError("Training seeds must be a subset of the prepared seen-partner protocol")
    for key in ("val_split", "positive_sampling", "positive_to_negative_ratio"):
        if options[key] != selection.get(key):
            raise ValueError(f"{key} differs from the prepared seen-partner protocol")
    split_ids = config["split_ids"] if split_ids is None else list(split_ids)
    if not split_ids or not set(split_ids).issubset(config['split_ids']) or len(set(split_ids)) != len(split_ids):
        raise ValueError("invalid split IDs")
    annotation_path = data / "mesh" / "extended_drug_info.csv"
    context = dict(experiment_id=config['experiment_id'], settings=options, arm=ARM,
        baseline_mode="local", scope_spec=asdict(SCOPE_BY_KEY['mid_level']),
        graph_hash=file_hash(graph_path), graph_variant="no_ddi", topo_settings=DENSE_GRAPH_SETTINGS,
        code_path_base="dtpkg", code_hashes={name: file_hash(PACKAGE_ROOT / name) for name in CODE_FILES},
        annotation_hash=file_hash(annotation_path))
    if context['graph_hash'] != config['source_hashes']['graph']:
        raise ValueError("graph differs from split preparation source")
    if context['annotation_hash'] != config['source_hashes']['annotations']:
        raise ValueError("annotations changed after split preparation")
    depth_map = load_drug_depths(annotation_path)
    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.device(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if threads is not None:
        torch.set_num_threads(threads)
    # Validate every selected prepared table before fitting even the first model.
    prepared = {split_id: load_prepared_split(prepared_dir / split_id) for split_id in split_ids}
    claim_output_directory(results_dir, dict(schema_version=1, kind="inductive_fits", context=context))
    all_predictions = []
    base_extractor = None
    for split_id in split_ids:
        manifest, emb, frames, test = prepared[split_id]
        if test.empty:
            raise ValueError(f"{split_id} has no evaluable test pairs")
        train_extractor = heldout_extractor = graph_factory = None
        for training_seed in training_seeds:
            destination = results_dir / split_id / f"seed_{training_seed}"
            signature = json_hash(dict(context=context, manifest=manifest, training_seed=training_seed,
                                       partition_hash=None))
            signature_file = destination / "fit_identity.json"
            if destination.exists() and any(destination.iterdir()) and not signature_file.exists():
                raise ValueError(f"Existing fit artifacts have no identity: {destination}")
            if signature_file.exists() and json.loads(signature_file.read_text())['signature'] != signature:
                raise ValueError(f"Saved fit settings differ: {destination}; use a new results directory")
            destination.mkdir(parents=True, exist_ok=True)
            write_json(signature_file, dict(signature=signature, context=context, split_id=split_id,
                training_seed=training_seed, manifest_hash=json_hash(manifest), partition=None))
            checkpoint = destination / "checkpoints" / "fit_00.pt"
            completed = destination / "completed.json"
            predictions_path = destination / "predictions.csv"
            if completed.exists():
                record = json.loads(completed.read_text())
                if (record['signature'] != signature or file_hash(predictions_path) != record['predictions_hash']
                        or file_hash(checkpoint) != record['checkpoint_hash']):
                    raise ValueError(f"Completed fit artifacts changed: {destination}")
                p = pd.read_csv(predictions_path)
                if "analysis_arm" not in p or not p.analysis_arm.eq("biological_only").all():
                    raise ValueError(f"Completed predictions have an inconsistent arm: {destination}")
                print(f"Reusing {split_id}, training seed {training_seed}")
            else:
                if checkpoint.exists():
                    print(f"Recovering predictions from saved checkpoint: {checkpoint}")
                    p = score_saved_fit(checkpoint, device, expected_identity=signature)
                    recover_ancillary_exports(checkpoint, destination)
                elif not fit_missing:
                    raise ValueError(f"Report-only mode: missing fit/checkpoint: {destination}")
                else:
                    if base_extractor is None:
                        base_extractor, provenance = make_extractor(variant="no_ddi", graph_path=graph_path,
                            backend="optimized", n_jobs=n_jobs, cache_dir=results_dir / "topo_cache")
                        provenance["arm"] = ARM
                        provenance["base_graph_path"] = str(graph_path)
                        write_json(results_dir / "topo_provenance.json", provenance)
                    if train_extractor is None:
                        train_extractor, heldout_extractor, graph_factory, checks = build_split_graphs(
                            base_extractor, manifest['heldout_drugs'])
                        write_json(results_dir / split_id / "graph_access_checks.json", checks)
                    p = fit_split_seed(manifest, emb, frames, test, depth_map, train_extractor,
                        heldout_extractor, graph_factory, options, training_seed, destination,
                        signature, device, fit_baseline=True)
                p = attach_metadata(p, manifest, training_seed, signature)
                p.to_csv(predictions_path, index=False)
                write_json(completed, dict(signature=signature, predictions_hash=file_hash(predictions_path),
                                           checkpoint_hash=file_hash(checkpoint), reference_baseline=None))
            all_predictions.append(p)
    if validate_experiment(prepared_dir, graph_path, data_dir=data) != config:
        raise ValueError("prepared experiment changed during fitting")
    if any(file_hash(PACKAGE_ROOT / name) != digest for name, digest in context['code_hashes'].items()):
        raise ValueError("training source code changed during fitting")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    from dtpkg.inductive.evaluation import scenario_metrics
    scenario_metrics(predictions).to_csv(results_dir / "scenario_metrics.csv", index=False)
    return predictions
