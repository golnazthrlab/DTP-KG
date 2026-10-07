"""Fail-fast checks for immutable inductive inputs, before any model fitting."""
from pathlib import Path
from dataclasses import asdict
import json

import joblib
import numpy as np
import pandas as pd

from dtpkg.project_paths import DATA_DIR
from dtpkg.mesh_scopes import SCOPE_BY_KEY
from dtpkg.evaluation_inputs import pair_drugs
from dtpkg.inductive.inductive_data import file_hash, json_hash, stratified_holdouts

REQUIRED_ARTIFACTS = frozenset(("embeddings.csv", "mesh_transform.joblib", "mesh_coverage.csv",
    "development_pairs.csv", "positive_pool_pairs.csv", "seen_unseen_pairs.csv", "unseen_unseen_pairs.csv",
    "pair_counts.csv"))


def validate_experiment(prepared_dir, graph_path, *, data_dir=None):
    """Verify the experiment identity and every declared current source.

    Synthetic unit tests may declare a smaller source dictionary. Production
    preparation declares all source entries, and each is checked, not just graph
    and annotations. Nothing is regenerated to repair a failed check.
    """
    prepared_dir = Path(prepared_dir)
    data = DATA_DIR if data_dir is None else Path(data_dir)
    config = json.loads((prepared_dir / "experiment.json").read_text())
    signature = {k: v for k, v in config.items() if k not in ("experiment_id", "split_ids")}
    if json_hash(signature) != config["experiment_id"]:
        raise ValueError("prepared experiment identity does not match its configuration")
    ids = config['split_ids']
    if not ids or len(ids) != len(set(ids)) or any(Path(x).name != x for x in ids):
        raise ValueError("invalid prepared split IDs")
    if len(ids) != len(config['seeds']) or ids != [f'split_{i:02d}' for i in range(len(config['seeds']))]:
        raise ValueError("split IDs must exactly match the declared split seeds")
    if 'scope_spec' in config and config['scope_spec'] != asdict(SCOPE_BY_KEY['mid_level']):
        raise ValueError("prepared scope settings differ from the current declared scope")
    sources = dict(graph=Path(graph_path), annotations=data/'mesh'/'extended_drug_info.csv',
        embedding_ids=data/'mesh'/'MeSH_mid_level_tfidf_svd128.csv',
        positives=data/'interactions'/'DDI_positive_pairs.csv',
        negatives=data/'interactions'/'DDI_negative_pairs.csv',
        drug_types=data/'drugbank'/'drugbank_drugs_release.csv')
    for key, digest in config['source_hashes'].items():
        if key not in sources:
            raise ValueError(f"unknown prepared source {key}")
        if file_hash(sources[key]) != digest:
            raise ValueError(f"prepared source changed: {key}")
    if config.get('source_code_base') == 'dtpkg':
        package_root = Path(__file__).resolve().parents[1]
        for name, digest in config['preparation_dependencies'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('invalid preparation source filename')
            if file_hash(package_root / relative) != digest:
                raise ValueError(f'preparation source changed: {name}')
    candidates = pd.read_csv(prepared_dir / 'candidates.csv').set_index('drug_id')
    # Reloading object-valued columns preserves preparation's normalized table.
    if json_hash(candidates.reset_index().to_dict('records')) != config['candidate_hash']:
        raise ValueError("candidate table differs from experiment manifest")
    roster = set(candidates.index)
    reconstructed = stratified_holdouts(candidates, config['fraction'], config['seeds'])
    seen_holdouts = set()
    for repeat, split_id in enumerate(ids):
        m = json.loads((prepared_dir / split_id / 'manifest.json').read_text())
        if m['experiment_id'] != config['experiment_id'] or m['split_id'] != split_id or m['split_repeat'] != repeat:
            raise ValueError("split manifest does not belong to the declared experiment")
        if m['split_seed'] != config['seeds'][repeat]:
            raise ValueError("split seed differs from prepared experiment")
        if set(m['heldout_drugs']) != set(reconstructed[repeat]['heldout_drugs']):
            raise ValueError("heldout IDs do not match the declared stratified sampling rule")
        h, d = m['heldout_drugs'], m['development_drugs']
        if len(set(h)) != len(h) or len(set(d)) != len(d) or set(h) & set(d) or set(h) | set(d) != roster:
            raise ValueError("drug split is not a disjoint partition of the candidate population")
        if (len(h), len(d), len(roster)) != (m['n_heldout'], m['n_development'], m['n_candidates']):
            raise ValueError("drug split size metadata disagrees with its identities")
        if tuple(sorted(h)) in seen_holdouts:
            raise ValueError("duplicate drug holdouts are not independent split repetitions")
        seen_holdouts.add(tuple(sorted(h)))
    return config


def validate_split_tables(folder, manifest, embeddings, frames):
    """Check embeddings, learned preprocessing IDs, pair labels and partitions."""
    folder = Path(folder)
    if not REQUIRED_ARTIFACTS.issubset(manifest['artifacts']):
        raise ValueError("prepared split lacks required artifact checksums")
    if not embeddings.index.is_unique or not embeddings.columns.is_unique or not np.isfinite(embeddings.to_numpy(float)).all():
        raise ValueError("invalid or nonfinite split embeddings")
    h, d = set(manifest['heldout_drugs']), set(manifest['development_drugs'])
    if not set(embeddings.index).issubset(h | d):
        raise ValueError("split embeddings contain drugs outside its population")
    transform = joblib.load(folder/'mesh_transform.joblib')  # trusted local prepared artifact
    if set(transform['heldout_ids']) != h or set(transform['development_ids']) != d or not set(transform['fit_ids']).issubset(d):
        raise ValueError("MeSH fit/transform identities disagree with the drug split")
    if transform['n_components'] != embeddings.shape[1] or transform['svd'].n_components != embeddings.shape[1]:
        raise ValueError("MeSH dimensions disagree with the fitted transform")
    identities = {}
    for name, frame in frames.items():
        if not frame.label.isin([0, 1]).all() or frame[['drug1','drug2','label']].isna().any().any():
            raise ValueError("invalid pair labels or missing drug IDs")
        a, b = frame.drug1.astype(str), frame.drug2.astype(str)
        if a.eq(b).any():
            raise ValueError("self pairs are not allowed")
        keys = pd.MultiIndex.from_arrays([np.minimum(a, b), np.maximum(a, b)])
        if keys.has_duplicates:
            raise ValueError("duplicate unordered pair in prepared table")
        identities[name] = set(keys)
        if not (a.isin(embeddings.index) & b.isin(embeddings.index)).all():
            raise ValueError("prepared pairs lack MeSH embeddings")
        count = a.isin(h).astype(int) + b.isin(h).astype(int)
        expected = {'development': 0, 'positive_pool': 0, 'seen_unseen': 1, 'unseen_unseen': 2}[name]
        if not count.eq(expected).all():
            raise ValueError("incorrect scenario assignment or held-out training endpoint")
    if not frames['positive_pool'].label.eq(1).all():
        raise ValueError("positive reservoir contains a nonpositive label")
    positives = frames['development'].query('label == 1')
    positive_keys = {tuple(sorted(x)) for x in positives[['drug1','drug2']].itertuples(index=False,name=None)}
    negative_keys = {tuple(sorted(x)) for x in frames['development'].query('label == 0')[['drug1','drug2']].itertuples(index=False,name=None)}
    if not positive_keys.issubset(identities['positive_pool']) or negative_keys & identities['positive_pool']:
        raise ValueError("development labels disagree with the positive reservoir")
    if frames['development'].label.nunique() != 2:
        raise ValueError("development cohort requires both classes for stratified validation")
    if identities['seen_unseen'] & identities['unseen_unseen']:
        raise ValueError("test scenarios overlap")


def load_prepared_split(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("invalid prepared artifact filename")
        if file_hash(folder / name) != digest:
            raise ValueError(f"Prepared split artifact changed: {folder / name}")
    emb = pd.read_csv(folder / "embeddings.csv", index_col=0)
    frames = {k: pd.read_csv(folder / f"{k}_pairs.csv")
              for k in ("development", "seen_unseen", "unseen_unseen", "positive_pool")}
    validate_split_tables(folder, manifest, emb, frames)
    test = pd.concat([frames[s] for s in ("seen_unseen", "unseen_unseen")], ignore_index=True)
    heldout = set(manifest["heldout_drugs"])
    if pair_drugs(frames['development']) & heldout or pair_drugs(frames['positive_pool']) & heldout:
        raise ValueError("held-out drug leaked into development pairs")
    for scenario, expected in (("seen_unseen", 1), ("unseen_unseen", 2)):
        p = frames[scenario]
        actual = p.drug1.isin(heldout).astype(int) + p.drug2.isin(heldout).astype(int)
        if not actual.eq(expected).all():
            raise ValueError("incorrect scenario assignment")
    return manifest, emb, frames, test
