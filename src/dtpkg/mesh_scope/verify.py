"""Independently verify saved scope predictions, partitions and checkpoint replay."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score
from dtpkg.ddi_labels import sha256
from dtpkg.data_loaders import build_pair_features
from dtpkg.models.mesh_mlp import MLP_DDI
from dtpkg.project_paths import PROJECT_ROOT, DATA_DIR


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _relative_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Expected a relative manifest path: {name}")
    return Path(root).joinpath(*relative.parts)


def _input_file(data_dir, name, path_base):
    if path_base == "data_dir":
        return _relative_file(data_dir, name)
    if path_base != "legacy":
        raise ValueError(f"Unknown input path base: {path_base}")
    relative = PurePosixPath(name)
    if not relative.parts or relative.parts[0] != "data":
        raise ValueError(f"Legacy study input is not under data/: {name}")
    return _relative_file(data_dir, str(PurePosixPath(*relative.parts[1:])))


def verify_source_provenance(out, expected, *, code_path_base="legacy"):
    """Check archived training sources while reporting later source changes.

    An incomplete archive cannot establish full training-source provenance.
    It can still support verification of saved metrics and model predictions.
    Without an archive, retain the original strict current-source requirement.
    """
    if code_path_base not in {"legacy", "dtpkg"}:
        raise ValueError(f"Unknown code path base: {code_path_base}")
    code_root = PACKAGE_ROOT if code_path_base == "dtpkg" else PROJECT_ROOT
    current = {path: sha256(_relative_file(code_root, path))
               if _relative_file(code_root, path).is_file() else None
               for path in expected}
    drift = {path: {'training_sha256': digest, 'current_sha256': current[path]}
             for path, digest in expected.items() if current[path] != digest}
    archive_path = Path(out) / 'source_snapshot.zip'
    if not archive_path.exists():
        if drift:
            raise ValueError(f"Current sources differ from the training run and no source archive exists: {list(drift)}")
        return dict(source_snapshot_present=False, original_source_snapshot_complete=None,
                    archived_source_hashes_match=None, archived_source_files=0,
                    missing_original_sources=[], current_source_drift={},
                    current_source_hashes_match=True)

    def require(condition, message):
        if not condition:
            raise ValueError(f"Invalid source_snapshot.zip: {message}")

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate archive entries')
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and
                    '\\' not in name and str(path) == name,
                    f'noncanonical archive path: {name}')
        require('snapshot_manifest.json' in names, 'missing snapshot_manifest.json')
        try:
            manifest = json.loads(archive.read('snapshot_manifest.json'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError('Invalid source_snapshot.zip: unreadable manifest') from exc
        require(isinstance(manifest, dict), 'manifest must be an object')
        require(manifest.get('expected_code_sha256') == expected,
                'expected hashes differ from the original run plan')
        captured = manifest.get('current_code_sha256')
        require(isinstance(captured, dict) and set(captured) == set(expected),
                'capture-time hashes must account for every original source')
        require(all(isinstance(digest, str) and len(digest) == 64 and
                    all(char in '0123456789abcdef' for char in digest)
                    for digest in captured.values()), 'malformed capture-time SHA256')
        missing = manifest.get('missing_original_sources')
        require(isinstance(missing, list) and all(isinstance(path, str) for path in missing),
                'missing_original_sources must be a list of paths')
        require(len(missing) == len(set(missing)) and set(missing) <= set(expected),
                'invalid or duplicate missing-source declarations')
        archived = set(names) - {'snapshot_manifest.json'}
        require(archived == set(expected) - set(missing),
                'archive entries do not match declared available original sources')
        for path in archived:
            digest = hashlib.sha256(archive.read(path)).hexdigest()
            require(digest == expected[path], f'archived source hash mismatch: {path}')
            require(captured[path] == digest, f'capture-time hash mismatch: {path}')
        require(all(captured[path] != expected[path] for path in missing),
                'a source declared unavailable matched its original at capture time')

    return dict(source_snapshot_present=True,
                original_source_snapshot_complete=not missing,
                archived_source_hashes_match=True, archived_source_files=len(archived),
                missing_original_sources=sorted(missing), current_source_drift=drift,
                current_source_hashes_match=not drift)


def verify(out, *, data_dir=None, report_path=None):
    """Read a private run without changing it; optionally write a separate report."""
    out = Path(out)
    data_dir = Path(data_dir or DATA_DIR).expanduser().resolve()
    plan = json.loads((out / 'run_plan.json').read_text())
    completed = json.loads((out / 'completion.json').read_text())
    expected_fits = plan['repeats'] * plan['folds'] * 3
    assert completed['status'] == 'complete' and completed['fits'] == expected_fits
    for path, digest in plan['input_sha256'].items():
        assert sha256(_input_file(data_dir, path, plan.get('input_path_base', 'legacy'))) == digest, path
    provenance = verify_source_provenance(out, plan['code_sha256'],
                                          code_path_base=plan.get('code_path_base', 'legacy'))
    logs = pd.read_csv(out / 'training_log.csv')
    assert len(logs) == expected_fits
    assert (logs.best_epoch <= logs.epochs_run).all()
    assert logs.groupby(['split_repeat', 'fold']).initial_weights_sha256.nunique().eq(1).all()
    schedules = pd.read_csv(out / 'sampling_audit.csv')
    assert schedules.groupby(['split_repeat', 'fold', 'epoch']).ordered_pairs_sha256.nunique().eq(1).all()
    assert len(schedules) == logs.epochs_run.sum()
    cohort = pd.read_csv(out / 'paired_cohort_pairs.csv')
    pred = pd.read_csv(out / 'predictions.csv')
    assert np.isfinite(pred.pred).all() and pred.pred.between(0, 1).all()
    splits = json.loads((out / 'split_manifest.json').read_text())
    test = pred[pred.phase == 'test']
    for (level, rep), p in test.groupby(['feature_set', 'split_repeat']):
        assert len(p) == len(cohort)
        assert not p.duplicated(['drug1', 'drug2']).any()
        pd.testing.assert_frame_equal(p[['drug1', 'drug2', 'label']].sort_values(['drug1', 'drug2']).reset_index(drop=True),
                                      cohort.sort_values(['drug1', 'drug2']).reset_index(drop=True))
    metric_frames = {'test': pd.read_csv(out / 'results_pair_bins_per_fold.csv'),
                     'validation': pd.read_csv(out / 'validation_metrics.csv')}
    for (phase, level, rep, fold), p in pred.groupby(['phase', 'feature_set', 'split_repeat', 'fold']):
        f = splits['repetitions'][rep][fold]
        idx = f['test_idx' if phase == 'test' else 'val_idx']
        pd.testing.assert_frame_equal(p[['drug1', 'drug2', 'label']].reset_index(drop=True),
                                      cohort.iloc[idx].reset_index(drop=True))
        metrics = metric_frames[phase]
        metrics = metrics[(metrics.feature_set == level) & (metrics.split_repeat == rep) & (metrics.fold == fold)]
        for row in metrics.itertuples():
            selected = p if row.pair_bin == 'overall' else p[p.pair_bin == row.pair_bin]
            assert len(selected) == row.n
            assert int(selected.label.sum()) == row.n_pos
            if selected.label.nunique() == 2:
                assert np.isclose(roc_auc_score(selected.label, selected.pred), row.auc, atol=1e-12)
            if len(selected):
                assert np.isclose(f1_score(selected.label, selected.pred >= .5, zero_division=0), row.f1, atol=1e-12)
    # Replay one restored model per scope, independently of the training loaders.
    replay_error = {}
    torch.set_num_threads(1)
    for level in logs.feature_set.unique():
        fit = out / 'fits/repeat_0_fold_0' / level
        state = torch.load(fit / 'checkpoint.pt', map_location='cpu', weights_only=True)
        emb = pd.read_csv(data_dir / f'mesh/MeSH_{level}_tfidf_svd128.csv', index_col=0)
        model = MLP_DDI(2 * emb.shape[1],
                        hidden_dims=tuple(plan['hidden_dims']), dropout=plan['dropout'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        p = test[(test.feature_set == level) & (test.split_repeat == 0) & (test.fold == 0)]
        x, _, _ = build_pair_features(emb, p)
        scaler = np.load(fit / 'scaler.npz')
        x = torch.tensor((x - scaler['mean']) / scaler['scale'], dtype=torch.float32)
        with torch.no_grad():
            replay = np.concatenate([model(batch).numpy() for batch in x.split(plan['batch_size'])])
        replay_error[level] = float(np.max(np.abs(replay - p.pred.to_numpy())))
        assert replay_error[level] < 1e-6
    comps = pd.read_csv(out / 'scope_comparisons_holm.csv')
    assert len(comps) == 72 and comps.family_size.eq(72).all()
    for row in comps.itertuples():
        table = metric_frames['test']
        a = table[(table.feature_set == row.model_a) & (table.pair_bin == row.category)].set_index(['split_repeat', 'fold'])[row.metric]
        b = table[(table.feature_set == row.model_b) & (table.pair_bin == row.category)].set_index(['split_repeat', 'fold'])[row.metric]
        assert np.isclose((a - b).mean(), row.estimate, equal_nan=True)
    result = dict(status='passed', fits=expected_fits, metric_rows_verified=sum(map(len, metric_frames.values())),
        predictions_verified=len(pred), checkpoint_replay_max_abs_error=replay_error,
        input_hashes_match=True,
        input_and_code_hashes_match=provenance['current_source_hashes_match'],
        paired_partitions_and_schedule_prefixes_match=True, **provenance)
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="private completed training-run directory")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report", type=Path, help="optional verification JSON output")
    args = parser.parse_args(argv)
    verify(args.run, data_dir=args.data_dir, report_path=args.report)


if __name__ == "__main__":
    main()
