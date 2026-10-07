"""Versioned drug holdouts and MeSH representations fitted without held-out drugs."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from dtpkg.mesh_scopes import SCOPE_BY_KEY, scope_table
from dtpkg.ddi_labels import POLICY
from dtpkg.data_loaders import _resolve_supervision_labels


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)


def guarded_pairs(pos_path, neg_path):
    pos, neg = _resolve_supervision_labels(pd.read_csv(pos_path), pd.read_csv(neg_path), POLICY, None)
    return pd.concat([pos.assign(label=1), neg.assign(label=0)], ignore_index=True)


def candidate_table(annotations, embedding_ids, graph, pairs):
    """Embedding-covered graph drugs with labels; strata use NON-DDI degree.

    Existing embedding row IDs define baseline availability only. Their fitted
    values are never reused. Split-specific vocabulary losses are audited later.
    Drug type is reported, not added to the depth x degree stratification.
    """
    depths = annotations.groupby("drugbank_id").knowledge_category
    if depths.nunique().gt(1).any():
        raise ValueError("conflicting MeSH knowledge categories")
    labeled = set(pairs.drug1) | set(pairs.drug2)
    graph_drugs = {str(n) for n, a in graph.nodes(data=True) if str(a.get("type", "")).lower() == "drug"}
    ids = sorted(set(map(str, embedding_ids)) & labeled & graph_drugs & set(annotations.drugbank_id))
    table = pd.DataFrame({"drug_id": ids}).set_index("drug_id")
    table["depth"] = depths.first().reindex(ids)
    table["non_ddi_degree"] = [sum(str(graph.nodes[v].get("type", "")).lower() != "drug"
                                     for v in graph.neighbors(d)) for d in ids]
    table["degree_bin"] = 0
    for _, group in table.groupby("depth"):
        # Duplicate cutpoints (including all-zero degrees) reduce bin count;
        # identical degrees are never artificially split by drug identity.
        bins = pd.qcut(group.non_ddi_degree, q=3, labels=False, duplicates="drop").fillna(0)
        table.loc[group.index, "degree_bin"] = bins.astype(int)
    table["stratum"] = table.depth + ":" + table.degree_bin.astype(str)
    return table


def stratified_holdouts(candidates, fraction=.15, seeds=(42, 43, 44, 45, 46)):
    """Independent overlapping holdouts with proportional, fixed-size quotas."""
    if not 0 < fraction < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("fraction must be in (0,1) and split seeds distinct")
    if not candidates.index.is_unique or candidates.empty:
        raise ValueError("unique, nonempty candidate drug table required")
    counts = candidates.groupby("stratum", sort=True).size()
    target = int(np.floor(len(candidates) * fraction + .5))
    if not 0 < target < len(candidates):
        raise ValueError("holdout must leave nonempty test and development sets")
    raw = counts * fraction
    quotas = np.floor(raw).astype(int)
    extra = target - int(quotas.sum())
    # Largest-remainder allocation avoids multiplying stratum proportions twice.
    for stratum in (raw - quotas).sort_values(ascending=False, kind="stable").index[:extra]:
        quotas.loc[stratum] += 1
    manifests = []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        heldout = []
        for stratum, group in candidates.groupby("stratum", sort=True):
            heldout.extend(rng.choice(sorted(group.index), size=int(quotas.loc[stratum]), replace=False).tolist())
        heldout = sorted(heldout)
        manifests.append(dict(split_id=f"split_{i:02d}", split_repeat=i, split_seed=int(seed),
            fraction_requested=fraction, n_candidates=len(candidates), n_heldout=len(heldout),
            n_development=len(candidates)-len(heldout), heldout_drugs=heldout,
            development_drugs=sorted(set(candidates.index)-set(heldout)),
            strata=[dict(stratum=str(k), n_population=int(counts[k]), n_heldout=int(quotas[k]))
                    for k in counts.index]))
    if len({tuple(m['heldout_drugs']) for m in manifests}) != len(manifests):
        raise ValueError("duplicate holdouts; choose a larger cohort or different seeds")
    return manifests


def _binary_matrix(rows, ids, vocabulary):
    drug_index = {d: i for i, d in enumerate(ids)}
    term_index = {t: i for i, t in enumerate(vocabulary)}
    selected = rows[rows.drugbank_id.isin(drug_index) & rows.tree_number.isin(term_index)]
    selected = selected.drop_duplicates(["drugbank_id", "tree_number"])
    return csr_matrix((np.ones(len(selected)), (selected.drugbank_id.map(drug_index),
                     selected.tree_number.map(term_index))), shape=(len(ids), len(vocabulary)))


def fit_inductive_mesh(annotations, development_ids, heldout_ids, scope_key="mid_level",
                       n_components=128, seed=42):
    """Fit term filtering, IDF and SVD on development drugs; transform holdouts.

    The unsupervised fit population is the declared outer-development drug set,
    including inner-validation drugs. No outer-heldout annotation affects the
    vocabulary, document frequencies, IDF or SVD basis. Fixed scope/frequency
    thresholds are inherited from MeSH scope selection, not retuned to holdout performance.
    """
    development_ids, heldout_ids = sorted(set(development_ids)), sorted(set(heldout_ids))
    if set(development_ids) & set(heldout_ids):
        raise ValueError("MeSH fit and heldout drugs overlap")
    scope = SCOPE_BY_KEY[scope_key]
    rows = scope_table(annotations, scope)
    train_rows = rows[rows.drugbank_id.isin(development_ids)]
    frequency = train_rows.groupby("tree_number").drugbank_id.nunique()
    vocab = sorted(frequency[(frequency >= scope.freq_min) & (frequency <= scope.freq_max)].index)
    if len(vocab) < 2:
        raise ValueError("fewer than two training-supported MeSH terms")
    all_ids = development_ids + heldout_ids
    binary = _binary_matrix(rows, all_ids, vocab)
    covered = np.asarray(binary.getnnz(axis=1)).ravel() > 0
    train_mask = covered[:len(development_ids)]
    fit_ids = np.asarray(development_ids)[train_mask].tolist()
    train_binary = binary[:len(development_ids)][train_mask]
    if min(train_binary.shape) < n_components:
        raise ValueError(f"MeSH training matrix {train_binary.shape} cannot support {n_components} components")
    tfidf = TfidfTransformer()
    train_tfidf = tfidf.fit_transform(train_binary)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(train_tfidf)
    # Transform all rows in the SAME fitted coordinate system, including train.
    values = svd.transform(tfidf.transform(binary[covered]))
    embeddings = pd.DataFrame(values, index=np.asarray(all_ids)[covered],
                              columns=[f"svd_{i+1}" for i in range(n_components)])
    coverage = pd.DataFrame(dict(drug_id=all_ids, heldout=[False]*len(development_ids)+[True]*len(heldout_ids),
                                has_retained_term=covered))
    state = dict(format_version=1, scope_key=scope_key, vocabulary=vocab, tfidf=tfidf, svd=svd,
                 development_ids=development_ids, heldout_ids=heldout_ids, fit_ids=fit_ids,
                 seed=seed, n_components=n_components)
    return embeddings, state, coverage


def balance_pairs(frame, seed):
    """Fixed test sampling, after eligibility/scenario assignment.

    Preserve a single-class group for explicit descriptive reporting; never
    fabricate negatives or remove the entire group to achieve nominal balance.
    """
    groups = [frame[frame.label == label] for label in (0, 1)]
    if any(g.empty for g in groups):
        return frame.copy().reset_index(drop=True)
    n = min(map(len, groups))
    return pd.concat([g.sample(n=n, random_state=seed) for g in groups], ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)


def split_pairs(pairs, embedding_ids, heldout_ids, seed=42):
    ids, heldout = set(embedding_ids), set(heldout_ids)
    eligible = pairs[pairs.drug1.isin(ids) & pairs.drug2.isin(ids)].copy()
    nheld = eligible.drug1.isin(heldout).astype(int) + eligible.drug2.isin(heldout).astype(int)
    pools = {name: eligible[nheld == n].reset_index(drop=True)
             for name, n in (("development", 0), ("seen_unseen", 1), ("unseen_unseen", 2))}
    selected = {name: balance_pairs(frame, seed) for name, frame in pools.items()}
    selected["positive_pool"] = pools["development"].query("label == 1").copy()
    counts = []
    for name in pools:
        for stage, frame in (("eligible", pools[name]), ("evaluated" if name != "development" else "development_cohort", selected[name])):
            present = set(frame.drug1) | set(frame.drug2)
            counts.append(dict(scenario=name, stage=stage, n_pairs=len(frame),
                n_pos=int(frame.label.sum()), n_neg=int((frame.label == 0).sum()),
                n_drugs=len(present), n_heldout_evaluated=len(present & heldout)))
    return selected, pd.DataFrame(counts)


def prepare_experiment(output_dir, annotations, candidates, pairs, source_hashes,
                       fraction=.15, seeds=(42,43,44,45,46), n_components=128):
    """Write immutable manifests, split-specific transforms and pair sets.

    Lower-level helper for synthetic validation or predeclared drug holdouts.
    The manuscript workflow uses prepare_seen_unseen.prepare.
    """
    output_dir = Path(output_dir)
    signature = dict(fraction=fraction, seeds=list(seeds), n_components=n_components,
                     source_hashes=source_hashes, code_hash=file_hash(__file__),
                     scope_spec=asdict(SCOPE_BY_KEY['mid_level']),
                     preparation_dependencies={name: file_hash(Path(__file__).resolve().parents[1] / name)
                         for name in ('mesh_scopes.py', 'data_loaders.py', 'ddi_labels.py')},
                     candidate_hash=json_hash(candidates.reset_index().to_dict("records")),
                     stratification="MeSH depth x within-depth non-DDI degree quantiles")
    experiment_id = json_hash(signature)
    config_path = output_dir / "experiment.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old["experiment_id"] != experiment_id:
            raise ValueError("Existing experiment differs; use a new output directory, preserving prior manifests")
        # An immutable completed preparation is read, never silently rewritten.
        saved = []
        saved_counts = []
        for split_id in old['split_ids']:
            manifest = json.loads((output_dir / split_id / 'manifest.json').read_text())
            if manifest['experiment_id'] != experiment_id:
                raise ValueError("saved split does not belong to prepared experiment")
            for name, digest in manifest['artifacts'].items():
                if file_hash(output_dir / split_id / name) != digest:
                    raise ValueError(f"prepared artifact changed: {split_id}/{name}")
            saved.append(manifest)
            saved_counts.append(pd.read_csv(output_dir / split_id / 'pair_counts.csv').assign(split_id=split_id))
        return saved, pd.concat(saved_counts, ignore_index=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("nonempty destination; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidates.csv")
    manifests = stratified_holdouts(candidates, fraction, seeds)
    all_counts, all_coverage = [], []
    for manifest in manifests:
        folder = output_dir / manifest["split_id"]
        folder.mkdir(exist_ok=True)
        emb, transformer, coverage = fit_inductive_mesh(annotations, manifest["development_drugs"],
            manifest["heldout_drugs"], n_components=n_components, seed=manifest["split_seed"])
        selected, counts = split_pairs(pairs, emb.index, manifest["heldout_drugs"], manifest["split_seed"])
        emb.to_csv(folder / "embeddings.csv")
        joblib.dump(transformer, folder / "mesh_transform.joblib")
        coverage.to_csv(folder / "mesh_coverage.csv", index=False)
        for name, frame in selected.items():
            frame.to_csv(folder / f"{name}_pairs.csv", index=False)
        counts.to_csv(folder / "pair_counts.csv", index=False)
        manifest.update(experiment_id=experiment_id,
            artifacts={p.name: file_hash(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != "manifest.json"})
        write_json(folder / "manifest.json", manifest)
        all_counts.append(counts.assign(split_id=manifest["split_id"]))
        all_coverage.append(coverage.assign(split_id=manifest["split_id"]))
    overlap = pd.DataFrame([[len(set(a['heldout_drugs']) & set(b['heldout_drugs'])) for b in manifests]
                            for a in manifests], index=[m['split_id'] for m in manifests],
                            columns=[m['split_id'] for m in manifests])
    overlap.to_csv(output_dir / "holdout_overlap_counts.csv")
    pd.concat(all_counts).to_csv(output_dir / "pair_counts.csv", index=False)
    pd.concat(all_coverage).to_csv(output_dir / "mesh_coverage.csv", index=False)
    write_json(config_path, dict(experiment_id=experiment_id, **signature,
                               split_ids=[m['split_id'] for m in manifests]))
    return manifests, pd.concat(all_counts, ignore_index=True)
