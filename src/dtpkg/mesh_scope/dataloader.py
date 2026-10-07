"""Common-coverage cohort and identical stratified partitions across scopes."""
import pandas as pd

from dtpkg.data_loaders import build_pair_features, _resolve_supervision_labels
from dtpkg.ddi_labels import POLICY


def common_scope_drugs(emb_dir, levels=("low_level", "mid_level", "deep_level")):
    """Drugs that have an embedding in *every* scope.

    The three scopes have different vocabularies and therefore different drug
    coverage (2,694 / 3,125 / 2,994 drugs). A comparison that is meant to isolate
    the effect of scope must hold the pairs fixed, so it can only use pairs whose
    drugs are covered in all scopes -- otherwise "scope" and "cohort" change
    together and per-(seed, fold) contrasts are not paired.
    """
    common = None
    for level in levels:
        ids = set(pd.read_csv(f"{emb_dir}/MeSH_{level}_tfidf_svd128.csv",
                              index_col=0, usecols=[0]).index.astype(str))
        common = ids if common is None else common & ids
    return common


def load_ddi_data(level: str, pos_path: str, neg_path: str, emb_dir: str, seed=42,
                  restrict_drugs=None, fixed_pairs=None):
    """Embeddings + balanced DDI pairs for one knowledge level.

    ``restrict_drugs``: optional drug-ID set; only pairs with both drugs in it
    (and in this scope's embedding) are eligible. ``fixed_pairs``: a
    ``drug1, drug2, label`` frame to use *as is* instead of sampling -- this is
    how :func:`load_paired_scope_data` gives every scope the identical pairs.
    The fixed negative cohort is matched by the same number of positives.
    Returns: X, y, pairs_df (aligned row-by-row).
    """
    emb_df = pd.read_csv(f"{emb_dir}/MeSH_{level}_tfidf_svd128.csv", index_col=0)
    emb_df.index = emb_df.index.astype(str)
    eligible = set(emb_df.index)
    if restrict_drugs is not None:
        eligible &= set(map(str, restrict_drugs))

    if fixed_pairs is not None:
        pairs_in = fixed_pairs[["drug1", "drug2", "label"]].copy()
        X, y, pairs = build_pair_features(emb_df, pairs_in)
        if len(pairs) != len(pairs_in):
            raise ValueError(f"[{level}] {len(pairs_in) - len(pairs)} fixed pairs lack an "
                             f"embedding in this scope; use common_scope_drugs().")
        print(f"[{level}] Using {len(pairs):,} fixed pairs ({int(sum(y))} pos, "
              f"{len(y)-int(sum(y))} neg) → feature shape {X.shape}")
        return X, y, pairs

    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    # Canonicalize and resolve labels before cohort construction or splitting.
    pos_df, neg_df = _resolve_supervision_labels(pos_df, neg_df, POLICY, None)
    pos_df["label"], neg_df["label"] = 1, 0

    pos_df = pos_df[pos_df["drug1"].isin(eligible) & pos_df["drug2"].isin(eligible)]
    neg_df = neg_df[neg_df["drug1"].isin(eligible) & neg_df["drug2"].isin(eligible)]
    n_pos_all, n_neg_all = len(pos_df), len(neg_df)
    print(f"\n[{level}] Eligible pairs" + (" (common-scope drugs)" if restrict_drugs is not None else "") + ":")
    print(f"   Positives: {n_pos_all:,}")
    print(f"   Negatives: {n_neg_all:,}")
    if n_pos_all == 0 or n_neg_all == 0:
        raise ValueError(f"[{level}] No positive or negative pairs found with embeddings!")

    if n_pos_all < n_neg_all:
        raise ValueError("Scope protocol requires at least as many eligible positives as negatives")
    n = n_neg_all
    pos_sample = pos_df.sample(n=n, random_state=seed)
    print(f"   ⚖️  Sampled {n:,} positives to match {n_neg_all:,} negatives.")
    all_pairs = pd.concat([pos_sample, neg_df], ignore_index=True)
    all_pairs = all_pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X, y, pairs = build_pair_features(emb_df, all_pairs)
    print(f"[{level}] Final dataset: {len(pairs):,} pairs ({int(sum(y))} pos, {len(y)-int(sum(y))} neg)")
    print(f"→ Feature shape: {X.shape}")
    return X, y, pairs


def load_paired_scope_data(levels, pos_path, neg_path, emb_dir, seed=42, n_splits=5):
    """One cohort, one fold assignment, three feature spaces.

    Samples the balanced pairs once over drugs covered by every scope, builds
    stratified outer folds once, and then materialises each scope's features for
    exactly those pairs. Per-(seed, fold) contrasts between scopes are then
    paired observations on identical test pairs. Returns
    ``(data, common_drugs)`` with ``data[level] = {X, y, folds, pairs}``.
    """
    common = common_scope_drugs(emb_dir, levels)
    X0, y0, pairs = load_ddi_data(levels[0], pos_path, neg_path, emb_dir, seed=seed,
                                  restrict_drugs=common)
    folds = create_folds(X0, y0, n_splits=n_splits, seed=seed)
    data = {levels[0]: {"X": X0, "y": y0, "folds": folds, "pairs": pairs}}
    for level in levels[1:]:
        X, y, p = load_ddi_data(level, pos_path, neg_path, emb_dir, seed=seed,
                                fixed_pairs=pairs)
        pd.testing.assert_frame_equal(p.reset_index(drop=True), pairs.reset_index(drop=True))
        data[level] = {"X": X, "y": y, "folds": folds, "pairs": p}
    print(f"\n[paired] {len(common):,} common drugs, {len(pairs):,} shared pairs, "
          f"{n_splits} shared stratified folds across {list(levels)}")
    return data, common


def create_folds(X, y, n_splits=5, seed=42, stratified=True):
    """Stratified outer folds by default (shared helper semantics)."""
    from dtpkg.data_loaders import create_folds as _shared
    return _shared(X, y, n_splits=n_splits, seed=seed, stratified=stratified)
