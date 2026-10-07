"""Prepare count-supported seen–unseen holdouts without inspecting model scores.

The first three eligible proposals in a declared seed sequence are retained.
Seen partners must occur in the first actual training epoch of every declared
training seed. Unseen–unseen pairs remain an unconstrained ancillary endpoint.
"""
from dataclasses import asdict
from pathlib import Path
import json


import joblib
import numpy as np
import pandas as pd

from dtpkg.ddi_sampling import EpochPositiveSampler
from dtpkg.evaluation_inputs import pair_drugs
from dtpkg.mesh_scopes import load_annotations, scope_table, SCOPE_BY_KEY
from dtpkg.project_paths import DATA_DIR
from dtpkg.topology_config import CANONICAL_GRAPH, load_graph
from dtpkg.inductive.inductive_data import (
    balance_pairs, candidate_table, file_hash, fit_inductive_mesh, guarded_pairs, json_hash,
    split_pairs, stratified_holdouts, write_json,
)
from dtpkg.inductive.preflight import planned_training_roster


DEPTHS = ("low_level", "mid_level", "deep_level")
CATEGORIES = tuple(f"{a}-{b}" for i, a in enumerate(DEPTHS) for b in DEPTHS[i:])


def exact_mesh_coverage(annotations, development_ids, candidate_ids):
    """Apply the exact development-only vocabulary filter before fitting SVD."""
    scope = SCOPE_BY_KEY["mid_level"]
    rows = scope_table(annotations, scope)
    frequency = rows[rows.drugbank_id.isin(development_ids)].groupby("tree_number").drugbank_id.nunique()
    vocabulary = frequency[(frequency >= scope.freq_min) & (frequency <= scope.freq_max)].index
    covered = set(rows.loc[rows.tree_number.isin(vocabulary), "drugbank_id"]) & set(candidate_ids)
    return covered, len(vocabulary)


def guaranteed_seen_roster(development, positive_pool, heldout_ids,
                           training_seeds=(101, 202, 303), val_split=.15):
    """Intersect exact epoch-zero endpoint sets across the declared training runs.

    The trainer excludes validation AND test pairs from the positive reservoir.
    Every test pair contains a held-out drug; this development-only reservoir
    contains none. Therefore excluding validation alone gives identical draws,
    and lets eligibility be established before selecting evaluation pairs.
    """
    heldout_ids = set(heldout_ids)
    if pair_drugs(development) & heldout_ids or pair_drugs(positive_pool) & heldout_ids:
        raise ValueError("held-out drug entered development pairs or positive reservoir")
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("distinct, nonempty training seeds required")
    rosters, audit = [], []
    for seed in training_seeds:
        _, train, validation = planned_training_roster(development, seed, val_split)
        train_pairs, validation_pairs = development.iloc[train], development.iloc[validation]
        sampler = EpochPositiveSampler(positive_pool, train_pairs.query("label == 0"),
            validation_pairs, seed=int(seed), fold=0, repetition=0,
            mode="per_epoch", positive_to_negative_ratio=1., excluded_drugs=heldout_ids)
        epoch = sampler.epoch(0)
        roster = pair_drugs(epoch)
        rosters.append(roster)
        audit.append(dict(training_seed=int(seed), n_epoch_zero_drugs=len(roster),
                          epoch_zero_pairs_hash=json_hash(epoch.to_dict("records")),
                          epoch_zero_drugs=sorted(roster)))
    return set.intersection(*rosters), audit


def annotate_seen_pairs(pairs, heldout_ids, depths):
    """Identify the held-out endpoint and unordered MeSH pair category."""
    heldout_ids = set(heldout_ids)
    a, b = pairs.drug1.isin(heldout_ids), pairs.drug2.isin(heldout_ids)
    if not (a ^ b).all():
        raise ValueError("seen-unseen support requires exactly one held-out endpoint")
    frame = pairs.copy()
    frame["heldout_drug"] = frame.drug1.where(a, frame.drug2)
    frame["seen_drug"] = frame.drug2.where(a, frame.drug1)
    order = {depth: i for i, depth in enumerate(DEPTHS)}
    frame["category"] = ["-".join(sorted((depths[x], depths[y]), key=order.__getitem__))
                         for x, y in zip(frame.drug1, frame.drug2)]
    return frame


def seen_support(frame):
    """Include empty cells, class-specific endpoint diversity and concentration."""
    records = []
    for category in CATEGORIES:
        for label in (0, 1):
            group = frame[frame.category.eq(category) & frame.label.eq(label)]
            frequencies = group.heldout_drug.value_counts()
            records.append(dict(category=category, label=label, n_pairs=len(group),
                n_heldout_drugs=len(frequencies), n_seen_drugs=group.seen_drug.nunique(),
                largest_heldout_share=float(frequencies.max() / len(group)) if len(group) else np.nan))
    return pd.DataFrame(records)


def select_supported_seen_pairs(pairs, heldout_ids, covered_ids, common_seen, depths, seed):
    """Apply coverage and guaranteed training exposure before category balancing."""
    eligible = pairs[pairs.drug1.isin(covered_ids) & pairs.drug2.isin(covered_ids)]
    one = eligible.drug1.isin(heldout_ids) ^ eligible.drug2.isin(heldout_ids)
    frame = annotate_seen_pairs(eligible[one], heldout_ids, depths)
    frame = frame[frame.seen_drug.isin(common_seen)].copy()
    selected = pd.concat([balance_pairs(group, seed) for _, group in frame.groupby("category")],
                         ignore_index=True) if len(frame) else frame.copy()
    return selected, seen_support(selected), frame


def support_passes(support, min_pairs=20, min_heldout=5):
    return bool(len(support) == 12 and support.n_pairs.ge(min_pairs).all()
                and support.n_heldout_drugs.ge(min_heldout).all())


def _pair_count(frame, heldout, scenario, stage):
    endpoints = pair_drugs(frame)
    return dict(scenario=scenario, stage=stage, n_pairs=len(frame),
                n_pos=int(frame.label.eq(1).sum()), n_neg=int(frame.label.eq(0).sum()),
                n_drugs=len(endpoints), n_heldout_evaluated=len(endpoints & set(heldout)))


def prepare(output_dir, *, data_dir=None, graph_path=None,
            source_dir=None, n_holdouts=3,
            proposal_seed=30000, max_proposals=200, training_seeds=(101, 202, 303),
            min_pairs=20, min_heldout=5, fraction=.15, n_components=128):
    from dtpkg.inductive.validation import validate_experiment
    from dtpkg.inductive.validation import load_prepared_split

    output_dir = Path(output_dir)
    data = DATA_DIR if data_dir is None else Path(data_dir)
    graph_path = Path(graph_path) if graph_path is not None else (
        CANONICAL_GRAPH if data_dir is None else data / "networks" / CANONICAL_GRAPH.name)
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("distinct, nonempty training seeds required")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    protocol = dict(name="seen_unseen_count_constrained_rejection_v1", primary_scenario="seen_unseen",
        n_holdouts=int(n_holdouts), proposal_seed_start=int(proposal_seed), max_proposals=int(max_proposals),
        min_pairs_per_class_per_depth_cell=int(min_pairs),
        min_heldout_drugs_per_class_per_depth_cell=int(min_heldout),
        training_seeds=list(training_seeds), val_split=.15, positive_sampling="per_epoch",
        positive_to_negative_ratio=1., seen_definition="intersection_of_training_epoch_zero_endpoints",
        pair_sampling="balance_separately_within_six_MeSH_pair_categories_after_seen_filter",
        selection_order="first_passing_proposals_in_ascending_seed_order",
        unseen_unseen="ancillary_original_global_balancing_without_support_constraints",
        no_model_scores_used=True)
    if n_holdouts < 1 or max_proposals < n_holdouts or min_pairs < 1 or min_heldout < 1:
        raise ValueError("positive support thresholds and sufficient proposal budget required")
    if (output_dir / "experiment.json").exists():
        config = validate_experiment(output_dir, graph_path, data_dir=data)
        if config.get("selection_protocol") != protocol or config["fraction"] != fraction or config["n_components"] != n_components:
            raise ValueError("requested preparation differs; use a new output directory")
        manifests = [load_prepared_split(output_dir / split_id)[0] for split_id in config["split_ids"]]
        return manifests, pd.read_csv(output_dir / "seen_unseen_support.csv")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("incomplete/nonempty destination; preserve it and use a new directory")
    paths = dict(annotations=data / "mesh" / "extended_drug_info.csv",
        embedding_ids=data / "mesh" / "MeSH_mid_level_tfidf_svd128.csv",
        positives=data / "interactions" / "DDI_positive_pairs.csv",
        negatives=data / "interactions" / "DDI_negative_pairs.csv", graph=graph_path,
        drug_types=data / "drugbank" / "drugbank_drugs_release.csv")
    annotations = load_annotations(paths["annotations"])
    pairs = guarded_pairs(paths["positives"], paths["negatives"])
    if source_dir is not None:
        source_dir = Path(source_dir)
        source = validate_experiment(source_dir, graph_path, data_dir=data)
        candidates = pd.read_csv(source_dir / "candidates.csv", index_col="drug_id")
    else:
        available = pd.read_csv(paths["embedding_ids"], index_col=0, usecols=[0]).index
        candidates = candidate_table(annotations, available, load_graph(graph_path), pairs)
        # Drug type is descriptive metadata; it does not affect holdout strata.
        drugs = pd.read_csv(paths["drug_types"], usecols=["drugbank_id", "type"], dtype=str)
        if drugs.groupby("drugbank_id").type.nunique().gt(1).any():
            raise ValueError("conflicting drug types for one drug ID")
        types = drugs.drop_duplicates("drugbank_id").set_index("drugbank_id").type.map(
            {"small molecule": "SM", "biotech": "Bio"}).fillna("Unknown")
        candidates["drug_type"] = candidates.index.map(types).fillna("Unknown")
        source = dict(source_hashes={name: file_hash(path) for name, path in paths.items()})
    pairs = pairs[pairs.drug1.isin(candidates.index) & pairs.drug2.isin(candidates.index)].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidates.csv")
    accepted, logs, proposal_support, per_drug, all_support, all_counts, all_coverage = [], [], [], [], [], [], []
    proposals = stratified_holdouts(candidates, fraction, tuple(range(proposal_seed, proposal_seed + max_proposals)))
    for manifest in proposals:
        seed, heldout = manifest["split_seed"], set(manifest["heldout_drugs"])
        covered, n_terms = exact_mesh_coverage(annotations, manifest["development_drugs"], candidates.index)
        selected, counts = split_pairs(pairs, covered, heldout, seed)
        common, rosters = guaranteed_seen_roster(selected["development"], selected["positive_pool"], heldout, training_seeds)
        seen, support, eligible_seen = select_supported_seen_pairs(pairs, heldout, covered, common, candidates.depth, seed)
        enough_dimensions = min(n_terms, len(covered - heldout)) >= n_components
        passed = support_passes(support, min_pairs, min_heldout) and enough_dimensions
        failed_cells = support[(support.n_pairs < min_pairs) | (support.n_heldout_drugs < min_heldout)]
        logs.append(dict(proposal_seed=seed, accepted=passed, n_common_seen_drugs=len(common),
            n_heldout_covered=len(heldout & covered), n_development_covered=len(covered - heldout),
            min_pairs=int(support.n_pairs.min()), min_heldout_contributors=int(support.n_heldout_drugs.min()),
            failed_cells=json.dumps(failed_cells[["category", "label", "n_pairs", "n_heldout_drugs"]].to_dict("records")),
            enough_svd_dimensions=enough_dimensions))
        proposal_support.append(support.assign(proposal_seed=seed, accepted=passed))
        pd.DataFrame(logs).to_csv(output_dir / "selection_proposals.csv", index=False)
        pd.concat(proposal_support).to_csv(output_dir / "selection_proposal_support.csv", index=False)
        print(f"Proposal {seed}: {'ACCEPT' if passed else 'reject'}; minimum class pairs={support.n_pairs.min()}, "
              f"held-out contributors={support.n_heldout_drugs.min()}", flush=True)
        if not passed:
            continue
        repeat = len(accepted)
        manifest.update(split_id=f"split_{repeat:02d}", split_repeat=repeat)
        split_id = manifest["split_id"]
        folder = output_dir / split_id
        folder.mkdir()
        embeddings, transformer, coverage = fit_inductive_mesh(annotations, manifest["development_drugs"],
            manifest["heldout_drugs"], n_components=n_components, seed=seed)
        if set(embeddings.index) != covered:
            raise ValueError("count-only MeSH eligibility disagrees with fitted representation")
        selected["seen_unseen"] = seen[["drug1", "drug2", "label"]].reset_index(drop=True)
        counts = counts[~(counts.scenario.eq("seen_unseen") & counts.stage.eq("evaluated"))]
        counts = pd.concat([counts, pd.DataFrame([
            _pair_count(eligible_seen, heldout, "seen_unseen", "guaranteed_seen_eligible"),
            _pair_count(seen, heldout, "seen_unseen", "evaluated")])], ignore_index=True)
        embeddings.to_csv(folder / "embeddings.csv")
        joblib.dump(transformer, folder / "mesh_transform.joblib")
        coverage.to_csv(folder / "mesh_coverage.csv", index=False)
        counts.to_csv(folder / "pair_counts.csv", index=False)
        for name, frame in selected.items():
            frame.to_csv(folder / f"{name}_pairs.csv", index=False)
        support.to_csv(folder / "seen_unseen_support.csv", index=False)
        write_json(folder / "guaranteed_seen_roster.json", dict(training_seeds=list(training_seeds),
            common_seen_drugs=sorted(common), first_epoch_rosters=rosters))
        manifest["artifacts"] = {p.name: file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}
        accepted.append(manifest)
        all_support.append(support.assign(split_id=split_id))
        all_counts.append(counts.assign(split_id=split_id))
        all_coverage.append(coverage.assign(split_id=split_id))
        per_drug.append(seen.groupby(["category", "label", "heldout_drug"]).size().rename("n_pairs").reset_index().assign(split_id=split_id))
        if len(accepted) == n_holdouts:
            break
    if len(accepted) != n_holdouts:
        raise ValueError(f"Only {len(accepted)} supported holdouts found in {max_proposals} proposals; diagnostic logs preserved")
    signature = dict(fraction=fraction, seeds=[m["split_seed"] for m in accepted], n_components=n_components,
        source_hashes=source["source_hashes"], code_hash=file_hash(__file__), selection_protocol=protocol,
        scope_spec=asdict(SCOPE_BY_KEY["mid_level"]), source_code_base="dtpkg",
        preparation_dependencies={name: file_hash(Path(__file__).resolve().parents[1] / name) for name in
            ("inductive/prepare_seen_unseen.py", "inductive/inductive_data.py", "inductive/preflight.py",
             "inductive/validation.py", "fusion/trainer.py", "evaluation_inputs.py",
             "ddi_sampling.py", "mesh_scopes.py", "data_loaders.py", "ddi_labels.py")},
        candidate_hash=json_hash(candidates.reset_index().to_dict("records")),
        stratification="MeSH depth x within-depth non-DDI degree quantiles")
    if source_dir is not None:
        signature["source_experiment_id"] = source["experiment_id"]
    experiment_id = json_hash(signature)
    for manifest in accepted:
        manifest["experiment_id"] = experiment_id
        write_json(output_dir / manifest["split_id"] / "manifest.json", manifest)
    pd.concat(all_support).to_csv(output_dir / "seen_unseen_support.csv", index=False)
    pd.concat(all_counts).to_csv(output_dir / "pair_counts.csv", index=False)
    pd.concat(all_coverage).to_csv(output_dir / "mesh_coverage.csv", index=False)
    pd.concat(per_drug).to_csv(output_dir / "seen_unseen_heldout_contributions.csv", index=False)
    overlap = pd.DataFrame([[len(set(a["heldout_drugs"]) & set(b["heldout_drugs"])) for b in accepted]
        for a in accepted], index=[m["split_id"] for m in accepted], columns=[m["split_id"] for m in accepted])
    overlap.to_csv(output_dir / "holdout_overlap_counts.csv")
    pair_sets = [set(zip(f.drug1, f.drug2)) for f in
                 (pd.read_csv(output_dir / m["split_id"] / "seen_unseen_pairs.csv") for m in accepted)]
    pd.DataFrame([[len(a & b) for b in pair_sets] for a in pair_sets], index=overlap.index,
                 columns=overlap.columns).to_csv(output_dir / "seen_unseen_pair_overlap_counts.csv")
    write_json(output_dir / "experiment.json", dict(experiment_id=experiment_id, **signature,
        split_ids=[m["split_id"] for m in accepted]))
    validate_experiment(output_dir, graph_path, data_dir=data)
    for manifest in accepted:
        load_prepared_split(output_dir / manifest["split_id"])
    print(f"Verified {len(accepted)} supported holdouts: {[m['split_seed'] for m in accepted]}", flush=True)
    return accepted, pd.concat(all_support, ignore_index=True)
