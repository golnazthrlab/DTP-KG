"""Read-only access to completed topology-only fits and their permitted graphs.

These iterators never train, fit a transform, rewrite an existing artifact, or
silently recover an incomplete run. Checkpoints are trusted local pickle files.
Each yielded graph reproduces the graph access of the saved feature extraction:
CV masks inner-validation and outer-test positives; biological-only induction
removes every held-out node for known-drug features and retains its biological
side information for held-out-drug features.
"""
from functools import lru_cache
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from dtpkg.network.graph_variants import graph_content_hash
from dtpkg.inductive.inductive_data import file_hash, json_hash


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=2)
def _read_graph_cached(path, size, mtime_ns):
    # File metadata participates in the cache key; a changed file is reloaded.
    graph = nx.read_graphml(path)
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("case studies require a simple undirected graph")
    return nx.freeze(graph), file_hash(path), graph_content_hash(graph)


def load_source_graph(path):
    """Return the shared base graph plus file/content hashes; do not mutate it."""
    path = Path(path).resolve()
    stat = path.stat()
    return _read_graph_cached(str(path), stat.st_size, stat.st_mtime_ns)


def _pair_table(frame):
    required = {"drug1", "drug2", "label"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("complete drug1/drug2/label columns required")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("binary pair labels required")
    out = frame[["drug1", "drug2", "label"]].copy()
    a, b = out.drug1.astype(str), out.drug2.astype(str)
    if a.eq(b).any():
        raise ValueError("self pairs are not permitted")
    out["pair_id"] = np.where(a < b, a + "|" + b, b + "|" + a)
    if out.pair_id.duplicated().any():
        raise ValueError("duplicate unordered pair in one fit")
    return out.set_index("pair_id").sort_index()


def _same_pairs(actual, expected, what):
    a, b = _pair_table(actual), _pair_table(expected)
    if not a.index.equals(b.index) or not np.array_equal(a.label.to_numpy(), b.label.to_numpy()):
        raise ValueError(f"{what}: saved pairs or labels differ")


def _positive_edges(*frames):
    return {tuple(sorted((str(a), str(b)))) for frame in frames
            for a, b in frame.loc[frame.label.eq(1), ["drug1", "drug2"]].itertuples(index=False, name=None)}


def _masked_graph(base, *frames, excluded_nodes=()):
    masked = _positive_edges(*frames)
    excluded = frozenset(excluded_nodes)
    return nx.subgraph_view(base, filter_node=lambda n: n not in excluded,
                            filter_edge=lambda a, b: tuple(sorted((str(a), str(b)))) not in masked)


def _graph_access_id(graph_hash, variant, frames, excluded_nodes=()):
    """Derived identity of access rules, not a recomputed graph-content hash."""
    return json_hash(dict(base_graph_sha256=graph_hash, variant=variant,
                          masked_positive_pairs=sorted(_positive_edges(*frames)),
                          removed_nodes=sorted(excluded_nodes)))


def _assert_targets_absent(graph, pairs):
    if any(graph.has_edge(str(a), str(b)) for a, b in pairs[["drug1", "drug2"]].itertuples(index=False, name=None)):
        raise ValueError("an evaluated pair remains an edge of the model-accessible graph")


def _validate_partition(bundle):
    tables = [bundle[name] for name in ("train_pairs", "validation_pairs", "test_pairs")]
    ids = [set(_pair_table(frame).index) for frame in tables]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("saved train/validation/test partitions overlap")


def _verify_topology_scores(bundle, predictions):
    """Re-score stored features without extracting topology or changing a fit."""
    _same_pairs(predictions, bundle["test_pairs"], "prediction/checkpoint alignment")
    if "topo_only" not in bundle["models"]:
        raise ValueError("checkpoint has no topology-only model")
    if predictions.empty or not np.isfinite(predictions.score.to_numpy(float)).all():
        raise ValueError("nonempty finite predictions required")
    if not predictions.score.between(0, 1).all() or not predictions.threshold.eq(.5).all():
        raise ValueError("expected neural probability scores and saved threshold 0.5")
    mu, sigma, columns = bundle["topo_only_normalization"]
    table = bundle["eval_topology"]
    if not table.index.is_unique or not np.isfinite(table[columns].to_numpy(float)).all():
        raise ValueError("saved topology table is invalid")
    arrays = [(table.loc[predictions[c].astype(str), columns].to_numpy(float) - mu) / sigma
              for c in ("drug1", "drug2")]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("saved topology normalization produces nonfinite features")
    model = bundle["models"]["topo_only"].cpu().eval()
    scores = []
    # Batched CPU evaluation differs from the original singleton/GPU path only
    # by floating-point roundoff. No randomness or fitting is used here.
    with torch.no_grad():
        for start in range(0, len(predictions), 512):
            features = [torch.as_tensor(a[start:start + 512], dtype=torch.float32) for a in arrays]
            scores.extend(torch.sigmoid(model(*features)).numpy().reshape(-1).tolist())
    differences = np.abs(np.asarray(scores) - predictions.score.to_numpy(float))
    if not np.allclose(scores, predictions.score.to_numpy(float), rtol=2e-5, atol=2e-6):
        raise ValueError(f"saved topology-only scores do not reproduce (max error {differences.max():.3g})")
    return float(differences.max())


def _subset(requested, available, name):
    values = list(available) if requested is None else list(requested)
    if not values or len(set(values)) != len(values) or not set(values).issubset(available):
        raise ValueError(f"invalid {name}: {values}")
    return values


def _checked_package_files(declared):
    """Check the installed release against a saved fit's relative code hashes."""
    if not isinstance(declared, dict) or not declared:
        raise ValueError("completed run lacks source-code hashes")
    checked = {}
    for name, expected in declared.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("invalid fitting-code path")
        source = PACKAGE_ROOT / path
        if file_hash(source) != expected:
            raise ValueError(f"current fitting code differs from completed run: {name}")
        checked[str(source)] = expected
    return checked


def _load_checkpoint(path):
    """Load only a trusted local checkpoint saved by this release."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        raise ValueError("checkpoint uses historical module names; use a completed "
                         "run made with this release, or redraw the retained aggregate tables") from error


def iter_transductive_fits(results_dir, graph_path,
                           repeats=None, folds=None, model="topo_only"):
    """Yield completed CV fits with original held-out scores and masked graphs.

    The historical CV bundles do not contain code hashes. Their saved split
    manifest, graph content provenance, and re-scored predictions are checked.
    If the gate-review audit exists, its original checkpoint/export hashes are
    checked too. All input hashes are included in the returned provenance.
    """
    if model != "topo_only":
        raise ValueError("these case studies use the separately trained topo_only model")
    results_dir = Path(results_dir)
    manifest_path = results_dir / "split_manifest.json"
    manifest = _json(manifest_path)
    repeats = _subset(repeats, range(manifest["num_repetitions"]), "CV repeats")
    folds = _subset(folds, range(manifest["num_folds"]), "CV folds")
    cohort = pd.DataFrame(manifest["pairs"])
    _pair_table(cohort)
    if len(cohort) != manifest["n_pairs"]:
        raise ValueError("CV cohort size differs from split manifest")
    predictions_path = results_dir / "oof_predictions.csv"
    predictions_hash = file_hash(predictions_path)
    all_rows = pd.read_csv(predictions_path)
    predictions = all_rows.loc[all_rows.model.eq(model)].copy()
    if set(predictions.inference_protocol) != {"cv"}:
        raise ValueError("expected CV prediction export")
    expected_groups = {(r, f) for r in range(manifest["num_repetitions"])
                       for f in range(manifest["num_folds"])}
    if set(predictions.groupby(["split_repeat", "fold"]).groups) != expected_groups:
        raise ValueError("CV export does not contain the complete declared fit grid")
    graph_provenance = _json(results_dir / "topo_provenance.json")
    base, graph_hash, content_hash = load_source_graph(graph_path)
    if graph_provenance.get("variant") != "full":
        raise ValueError("transductive case study expects the saved full-graph arm")
    if graph_provenance.get("graph_content_hash") != content_hash:
        raise ValueError("source graph differs from the graph used for the saved CV fits")
    audit_path = results_dir / "gate_review/analysis_provenance.json"
    audit = _json(audit_path) if audit_path.exists() else None
    audited_hashes = {} if audit is None else {Path(v["file"]).name: v["sha256"] for v in audit["checkpoints"]}
    if audit is not None and audit["oof_sha256"] != predictions_hash:
        raise ValueError("CV prediction export differs from its completed gate-review audit")
    source_files = {str(Path(graph_path).resolve()): graph_hash,
                    str(predictions_path.resolve()): predictions_hash,
                    str(manifest_path.resolve()): file_hash(manifest_path),
                    str((results_dir / "topo_provenance.json").resolve()): file_hash(results_dir / "topo_provenance.json")}
    plan_path = results_dir / "run_plan.json"
    if plan_path.is_file():
        plan = _json(plan_path)
        if plan.get("code_path_base") == "dtpkg":
            complete_path = results_dir / "completion.json"
            completion = _json(complete_path)
            if completion.get("status") != "complete":
                raise ValueError("transductive source run is incomplete")
            if (plan.get("repeats"), plan.get("folds")) != (manifest["num_repetitions"], manifest["num_folds"]):
                raise ValueError("transductive run plan differs from saved split grid")
            if plan.get("graph_file_sha256") != graph_hash:
                raise ValueError("transductive source graph differs from completed run")
            source_files.update(_checked_package_files(plan["code_sha256"]))
            for path in (plan_path, complete_path):
                source_files[str(path.resolve())] = file_hash(path)
    # CV bundles predate fit-code identities. Record the current model code
    # used for verified score replay so a cached report cannot hide its change.
    for name in ("fusion/topology_baseline.py", "fusion/trainer.py"):
        source_files[str(PACKAGE_ROOT / name)] = file_hash(PACKAGE_ROOT / name)
    if audit is not None:
        source_files[str(audit_path.resolve())] = file_hash(audit_path)
    for repeat in repeats:
        for fold in folds:
            checkpoint_path = results_dir / f"checkpoints/fit_{repeat:02d}_{fold:02d}.pt"
            checkpoint_hash = file_hash(checkpoint_path)
            if audit is not None and audited_hashes.get(checkpoint_path.name) != checkpoint_hash:
                raise ValueError(f"CV checkpoint differs from completed audit: {checkpoint_path}")
            bundle = _load_checkpoint(checkpoint_path)
            if bundle.get("protocol") != "cv" or bundle.get("format_version") != 1:
                raise ValueError("unsupported CV checkpoint")
            _validate_partition(bundle)
            split = manifest["repetitions"][repeat][fold]
            for key, indices in (("train_pairs", "train_idx"), ("validation_pairs", "val_idx"),
                                 ("test_pairs", "test_idx")):
                _same_pairs(bundle[key], cohort.iloc[split[indices]], f"CV {key}/manifest")
            rows = predictions.loc[predictions.split_repeat.eq(repeat) & predictions.fold.eq(fold)].copy()
            for key in ("split_repeat", "fold", "training_seed", "n_outer_train", "n_outer_test"):
                if not rows[key].eq(bundle["metadata"][key]).all():
                    raise ValueError(f"CV prediction/checkpoint metadata differ: {key}")
            error = _verify_topology_scores(bundle, rows)
            fit_id = f"transductive_repeat_{repeat:02d}_fold_{fold:02d}"
            rows = rows.assign(fit_id=fit_id, setting="transductive", scenario="transductive",
                               split_id=f"repeat_{repeat:02d}", drug1_heldout=False, drug2_heldout=False)
            graph = _masked_graph(base, bundle["validation_pairs"], bundle["test_pairs"])
            _assert_targets_absent(graph, bundle["test_pairs"])
            _assert_targets_absent(graph, bundle["validation_pairs"])
            graph_id = _graph_access_id(graph_hash, "full", [bundle["validation_pairs"], bundle["test_pairs"]])
            provenance = dict(fit_id=fit_id, setting="transductive", model=model,
                source_predictions=str(predictions_path), predictions_sha256=predictions_hash,
                checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                split_manifest_sha256=file_hash(manifest_path), graph_sha256=graph_hash,
                graph_content_hash=content_hash, graph_variant="full",
                feature_settings=graph_provenance["feature_settings"],
                masked_positive_pairs=len(_positive_edges(bundle["validation_pairs"], bundle["test_pairs"])),
                score_reproduction_max_absolute_error=error, training_performed=False,
                source_audit_checked=audit is not None,
                train_graph_id=graph_id, inference_graph_id=graph_id,
                graph_id_kind="derived_from_base_hash_variant_positive_mask_and_removed_nodes",
                source_files={**source_files, str(checkpoint_path.resolve()): checkpoint_hash})
            yield dict(predictions=rows, train_graph=graph, inference_graph=graph,
                       heldout_drugs=frozenset(), fit_id=fit_id, provenance=provenance,
                       train_graph_id=graph_id, inference_graph_id=graph_id)


def iter_inductive_fits(results_dir, prepared_dir, graph_path, *, data_dir,
                        split_ids=None, training_seeds=(101, 202, 303), model="topo_only"):
    """Yield completed biological-only fits, with exact endpoint graph access.

    ``results_dir`` is the experiment-05 CLI output root, or its ``fits``
    child containing ``split_*/seed_*``.
    All declared holdouts are selected; this need not equal the historical
    five-holdout snapshot used for the published figure.
    The primary biological-only arm is deliberately required: a known-DDI arm
    needs its separately verified context partition and must not inherit these
    graph-access assumptions.
    """
    if model != "topo_only":
        raise ValueError("these case studies use the separately trained topo_only model")
    from dtpkg.inductive.inductive_experiments import ARM
    from dtpkg.inductive.validation import load_prepared_split, validate_experiment
    from dtpkg.inductive.inductive_fit import CODE_FILES
    results_dir, prepared_dir = Path(results_dir), Path(prepared_dir)
    data_dir = Path(data_dir).expanduser().resolve()
    experiment = validate_experiment(prepared_dir, graph_path, data_dir=data_dir)
    split_ids = _subset(split_ids, experiment["split_ids"], "inductive split IDs")
    if (results_dir / "fits").is_dir():
        if any((results_dir / split_id).exists() for split_id in split_ids):
            raise ValueError("ambiguous inductive results: both direct and nested fit directories exist")
        results_dir = results_dir / "fits"
    training_seeds = list(training_seeds)
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("distinct nonempty training seeds required")
    base, graph_hash, content_hash = load_source_graph(graph_path)
    if graph_hash != experiment["source_hashes"]["graph"]:
        raise ValueError("source graph differs from prepared inductive experiment")
    # Materialize only biological edges once. The resulting graph has about
    # 286k edges, and all later endpoint/split restrictions are read-only views.
    drugs = {n for n, d in base.nodes(data=True) if str(d.get("type", "")).lower() == "drug"}
    biological = nx.Graph()
    biological.add_nodes_from(base.nodes(data=True))
    biological.add_edges_from((a, b, d) for a, b, d in base.edges(data=True)
                              if not (a in drugs and b in drugs))
    current_code = {name: file_hash(PACKAGE_ROOT / name) for name in CODE_FILES}
    source_paths = dict(graph=Path(graph_path), annotations=data_dir / "mesh/extended_drug_info.csv",
        embedding_ids=data_dir / "mesh/MeSH_mid_level_tfidf_svd128.csv",
        positives=data_dir / "interactions/DDI_positive_pairs.csv",
        negatives=data_dir / "interactions/DDI_negative_pairs.csv",
        drug_types=data_dir / "drugbank/drugbank_drugs_release.csv")
    source_files = {str(Path(source_paths[name]).resolve()): digest
                    for name, digest in experiment["source_hashes"].items()}
    source_files.update({str((PACKAGE_ROOT / name).resolve()): digest for name, digest in current_code.items()})
    for name in ("experiment.json", "candidates.csv"):
        source_files[str((prepared_dir / name).resolve())] = file_hash(prepared_dir / name)
    for split_id in split_ids:
        manifest, _, frames, test = load_prepared_split(prepared_dir / split_id)
        heldout = frozenset(manifest["heldout_drugs"])
        split_source_files = {str((prepared_dir / split_id / name).resolve()): digest
                              for name, digest in manifest["artifacts"].items()}
        manifest_path = prepared_dir / split_id / "manifest.json"
        split_source_files[str(manifest_path.resolve())] = file_hash(manifest_path)
        for seed in training_seeds:
            folder = results_dir / split_id / f"seed_{seed}"
            identity = _json(folder / "fit_identity.json")
            completion = _json(folder / "completed.json")
            context = identity["context"]
            expected_arm = ARM
            if context.get("arm") != expected_arm or context.get("graph_variant") != "no_ddi":
                raise ValueError("inductive case study requires the biological_only arm")
            if context.get("code_path_base") != "dtpkg" or context.get("code_hashes") != current_code:
                raise ValueError("current release fitting code differs from the completed inductive fit; "
                                 "use a completed run made with this release")
            if context.get("graph_hash") != graph_hash or context.get("experiment_id") != experiment["experiment_id"]:
                raise ValueError("inductive fit source graph/experiment mismatch")
            if context.get("annotation_hash") != experiment["source_hashes"]["annotations"]:
                raise ValueError("inductive fit annotation source mismatch")
            signature = json_hash(dict(context=context, manifest=manifest, training_seed=seed, partition_hash=None))
            if (identity["signature"] != signature or completion["signature"] != signature
                    or identity["manifest_hash"] != json_hash(manifest)
                    or identity["split_id"] != split_id or identity["training_seed"] != seed):
                raise ValueError(f"inductive fit identity mismatch: {folder}")
            predictions_path, checkpoint_path = folder / "predictions.csv", folder / "checkpoints/fit_00.pt"
            predictions_hash, checkpoint_hash = file_hash(predictions_path), file_hash(checkpoint_path)
            if predictions_hash != completion["predictions_hash"] or checkpoint_hash != completion["checkpoint_hash"]:
                raise ValueError(f"completed inductive fit artifact changed: {folder}")
            bundle = _load_checkpoint(checkpoint_path)
            if bundle.get("protocol") != "inductive_fixed_holdout" or bundle.get("fit_identity") != signature:
                raise ValueError("checkpoint protocol/identity mismatch")
            if set(bundle["heldout_drugs"]) != heldout:
                raise ValueError("checkpoint/prepared held-out drug sets differ")
            _validate_partition(bundle)
            _same_pairs(bundle["test_pairs"], test, "inductive checkpoint/prepared test pairs")
            _same_pairs(pd.concat([bundle["train_pairs"], bundle["validation_pairs"]]),
                        frames["development"], "inductive checkpoint/prepared development pairs")
            rows = pd.read_csv(predictions_path)
            rows = rows.loc[rows.model.eq(model)].copy()
            expected_metadata = dict(analysis_arm="biological_only", source_arm="biological_only",
                fit_identity=signature, split_id=split_id, training_seed=seed,
                split_repeat=manifest["split_repeat"], inference_protocol="inductive_repeated_drug_holdout")
            for key, value in expected_metadata.items():
                if key not in rows or not rows[key].eq(value).all():
                    raise ValueError(f"inductive prediction metadata differ: {key}")
            nheld = rows.drug1.isin(heldout).astype(int) + rows.drug2.isin(heldout).astype(int)
            if not nheld.isin([1, 2]).all() or not rows.scenario.eq(np.where(nheld.eq(1), "seen_unseen", "unseen_unseen")).all():
                raise ValueError("inductive scenarios disagree with held-out endpoint identities")
            for endpoint in ("drug1", "drug2"):
                if not rows[endpoint + "_heldout"].eq(rows[endpoint].isin(heldout)).all():
                    raise ValueError("inductive held-out flags disagree with prepared split")
            error = _verify_topology_scores(bundle, rows)
            train_graph = _masked_graph(biological, bundle["validation_pairs"], excluded_nodes=heldout)
            inference_graph = _masked_graph(biological, bundle["validation_pairs"])
            for graph in (train_graph, inference_graph):
                _assert_targets_absent(graph, bundle["test_pairs"])
                _assert_targets_absent(graph, bundle["validation_pairs"])
            train_graph_id = _graph_access_id(graph_hash, "no_ddi", [bundle["validation_pairs"]], heldout)
            inference_graph_id = _graph_access_id(graph_hash, "no_ddi", [bundle["validation_pairs"]])
            fit_id = f"inductive_{split_id}_seed_{seed}"
            rows = rows.assign(fit_id=fit_id, setting="inductive")
            provenance = dict(fit_id=fit_id, setting="inductive", model=model, analysis_arm="biological_only",
                source_predictions=str(predictions_path), predictions_sha256=predictions_hash,
                checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                fit_identity=signature, prepared_manifest_sha256=file_hash(prepared_dir / split_id / "manifest.json"),
                graph_sha256=graph_hash, graph_content_hash=content_hash, graph_variant="no_ddi",
                feature_settings=context["topo_settings"], n_heldout_drugs=len(heldout),
                known_endpoint_graph="biological graph without all held-out nodes",
                heldout_endpoint_graph="biological graph retaining side information for held-out nodes",
                score_reproduction_max_absolute_error=error, training_performed=False,
                train_graph_id=train_graph_id, inference_graph_id=inference_graph_id,
                graph_id_kind="derived_from_base_hash_variant_positive_mask_and_removed_nodes",
                source_files={**source_files, **split_source_files,
                    str((folder / "fit_identity.json").resolve()): file_hash(folder / "fit_identity.json"),
                    str((folder / "completed.json").resolve()): file_hash(folder / "completed.json"),
                    str(predictions_path.resolve()): predictions_hash,
                    str(checkpoint_path.resolve()): checkpoint_hash})
            yield dict(predictions=rows, train_graph=train_graph, inference_graph=inference_graph,
                       heldout_drugs=heldout, fit_id=fit_id, provenance=provenance,
                       train_graph_id=train_graph_id, inference_graph_id=inference_graph_id)
