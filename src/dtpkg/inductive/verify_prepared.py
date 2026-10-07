"""Replay prepared MeSH transformations and verify fixed splits without training."""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from dtpkg.inductive.inductive_data import _binary_matrix, file_hash, write_json
from dtpkg.inductive.validation import load_prepared_split
from dtpkg.inductive.validation import validate_experiment
from dtpkg.mesh_scopes import load_annotations, scope_table, SCOPE_BY_KEY
from dtpkg.project_paths import DATA_DIR
from dtpkg.topology_config import CANONICAL_GRAPH


def verify_prepared(prepared_dir, output=None, *, data_dir=None, graph_path=None):
    prepared_dir = Path(prepared_dir)
    data = DATA_DIR if data_dir is None else Path(data_dir)
    graph_path = Path(graph_path) if graph_path is not None else (
        CANONICAL_GRAPH if data_dir is None else data / "networks" / CANONICAL_GRAPH.name)
    config = validate_experiment(prepared_dir, graph_path, data_dir=data)
    annotations = load_annotations(data / "mesh" / "extended_drug_info.csv")
    checks = []
    for split_id in config['split_ids']:
        folder = prepared_dir / split_id
        m, embeddings, frames, test = load_prepared_split(folder)
        state = joblib.load(folder / 'mesh_transform.joblib')
        scope = SCOPE_BY_KEY[state['scope_key']]
        rows = scope_table(annotations, scope)
        development = state['development_ids']
        heldout = state['heldout_ids']
        frequency = rows[rows.drugbank_id.isin(development)].groupby('tree_number').drugbank_id.nunique()
        vocab = sorted(frequency[frequency.between(scope.freq_min, scope.freq_max)].index)
        if vocab != state['vocabulary']:
            raise ValueError('saved vocabulary disagrees with development-only term filtering')
        binary = _binary_matrix(rows, development + heldout, vocab)
        covered = np.asarray(binary.getnnz(axis=1)).ravel() > 0
        fit_mask = covered[:len(development)]
        fit_ids = np.array(development)[fit_mask].tolist()
        if fit_ids != state['fit_ids'] or set(fit_ids) & set(heldout):
            raise ValueError('saved MeSH fit IDs include a heldout drug or omit a supported development row')
        fit_binary = binary[:len(development)][fit_mask]
        expected_idf = np.log((1 + len(fit_ids)) / (1 + np.asarray(fit_binary.sum(axis=0)).ravel())) + 1
        np.testing.assert_allclose(state['tfidf'].idf_, expected_idf, rtol=1e-12, atol=1e-12)
        ids = np.array(development + heldout)[covered]
        if list(embeddings.index) != list(ids):
            raise ValueError('saved embedding rows disagree with split-specific term coverage')
        replay = state['svd'].transform(state['tfidf'].transform(binary[covered]))
        error = float(np.max(np.abs(embeddings.to_numpy() - replay)))
        np.testing.assert_allclose(embeddings, replay, rtol=1e-10, atol=1e-12)
        checks.append(dict(split_id=split_id, n_heldout=len(heldout), n_fit_drugs=len(fit_ids),
            n_heldout_with_embeddings=int(covered[len(development):].sum()), n_terms=len(vocab),
            max_embedding_replay_error=error, test_pairs=len(test),
            manifest_sha256=file_hash(folder / 'manifest.json')))
    result = dict(status='passed', no_model_fitting=True, experiment_id=config['experiment_id'],
        checks=checks, verifier_sha256=file_hash(__file__))
    if output is not None:
        output = Path(output)
        if output.resolve().is_relative_to(prepared_dir.resolve()):
            raise ValueError('verification output must be outside immutable prepared inputs')
        if output.exists():
            raise FileExistsError(f'verification output already exists: {output}')
        write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--graph-path', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_prepared(args.prepared, args.output, data_dir=args.data_dir, graph_path=args.graph_path), indent=2))
