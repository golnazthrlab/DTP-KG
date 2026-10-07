import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch

from sklearn.model_selection import StratifiedKFold

from dtpkg.ddi_labels import (POLICY, QUARANTINED_PAIRS, check_supervision_files,
                        format_report, load_release_pairs, resolve_labels)


class DDIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_pair_features(emb_df, pairs_df, pair_mode="sym"):
    """Symmetric per-drug embeddings for each pair, aligned with ``pairs``.

    Vectorised: pairs whose drugs both have an embedding row are selected with
    boolean masks and gathered with one ``take`` per side. Build this only for
    the pairs that will actually be used -- the eligible positive pool is now
    ~500k pairs, and materialising 256-float vectors for all of them to train on
    a few thousand was the loader's dominant cost.
    """
    if pair_mode not in ("sym", "concat"):
        raise ValueError("pair_mode must be 'sym' or 'concat'")
    if len(pairs_df) == 0:
        return (np.empty((0, 2 * emb_df.shape[1])), np.empty(0),
                pd.DataFrame(columns=["drug1", "drug2", "label"]))
    index = pd.Index(emb_df.index.astype(str))
    d1 = pairs_df["drug1"].astype(str).to_numpy()
    d2 = pairs_df["drug2"].astype(str).to_numpy()
    i1 = index.get_indexer(d1)
    i2 = index.get_indexer(d2)
    keep = (i1 >= 0) & (i2 >= 0)
    missing = int((~keep).sum())
    if missing > 0:
        print(f"Skipped {missing} pairs with missing embeddings.")
    values = emb_df.to_numpy()
    a, b = values[i1[keep]], values[i2[keep]]
    X = np.concatenate([a + b, np.abs(a - b)] if pair_mode == "sym" else [a, b], axis=1)
    y = pairs_df["label"].to_numpy()[keep]
    pairs = pd.DataFrame({"drug1": d1[keep], "drug2": d2[keep], "label": y})
    return X, y, pairs


def _resolve_supervision_labels(pos_df, neg_df, conflict_policy, reference_pairs_path):
    """Canonical label guard shared by the experiment loaders.

    Historically the loader sampled and split *rows*: the positive file listed
    many pairs in both orientations, the negative file had exact duplicates, and
    2,057 of the 5,404 unique reliable-negative pairs (Zheng et al. 2019) are
    documented interactions in the supplied DrugBank export. Pairs are therefore
    reduced to their unordered identity; a conflicting pair is handled under
    ``conflict_policy`` (default: a documented interaction, so positive); the
    four quarantined pairs are dropped from both classes; and the two classes
    are asserted disjoint -- all *before* any filtering, balancing or fold
    assignment. On the locally prepared study interaction files this finds nothing
    and changes nothing; it exists so that other inputs cannot silently
    reintroduce contradictory labels.
    """
    reference = load_release_pairs(reference_pairs_path) if reference_pairs_path else None
    pos_df, neg_df, report = resolve_labels(pos_df, neg_df, reference, policy=conflict_policy)
    print(format_report(report))
    if report["conflicts"]["total"] and conflict_policy == "keep":
        print("   ⚠️  conflicting labels retained (policy='keep'); do not train on this.")
    quarantined = set(QUARANTINED_PAIRS)
    for name, df in (("positives", pos_df), ("negatives", neg_df)):
        hit = [k in quarantined for k in zip(df["drug1"], df["drug2"])]
        if any(hit):
            print(f"   dropped {sum(hit)} quarantined pair(s) from the {name}")
            df.drop(df.index[hit], inplace=True)
    pos_df = pos_df.reset_index(drop=True)
    neg_df = neg_df.reset_index(drop=True)
    problems = check_supervision_files(pos_df, neg_df, reference_df=reference)
    if problems:
        raise ValueError(f"supervision files fail the label guard: {problems}")
    return pos_df, neg_df


def load_ddi_data(level: str, pos_path: str, neg_path: str, emb_dir: str,
                  seed=42, max_diff=500, inductive_drugs=None,
                  balance=True, conflict_policy=POLICY, reference_pairs_path=None,
                  return_positive_pool=False):
    """Balanced (or full) labelled pair set for one MeSH scope.

    ``conflict_policy`` / ``reference_pairs_path``: see the R1.2 guard below.
    ``reference_pairs_path`` may point at ``ddi_labels.RELEASE_DDI`` to also
    check against the complete DrugBank release; the resolved on-disk files
    already account for it, so the default is None.
    """

    np.random.seed(seed)

    # ----------------------------------------------------
    # 1. Load embeddings
    # ----------------------------------------------------
    emb_path = f"{emb_dir}/MeSH_{level}_tfidf_svd128.csv"
    emb_df = pd.read_csv(emb_path, index_col=0)
    emb_drugs = set(emb_df.index)

    # ----------------------------------------------------
    # 2. Load raw positive/negative DDI files
    # ----------------------------------------------------
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    pos_df, neg_df = _resolve_supervision_labels(
        pos_df, neg_df, conflict_policy, reference_pairs_path)
    pos_df["label"], neg_df["label"] = 1, 0

    # ----------------------------------------------------
    # 3. Remove inductive test drugs if needed (inductive mode)
    # ----------------------------------------------------
    if inductive_drugs:
        print(f"🔍 Inductive mode: excluding {len(inductive_drugs)} test drugs.")
        pos_df = pos_df[
            (~pos_df["drug1"].isin(inductive_drugs)) &
            (~pos_df["drug2"].isin(inductive_drugs))
        ]
        neg_df = neg_df[
            (~neg_df["drug1"].isin(inductive_drugs)) &
            (~neg_df["drug2"].isin(inductive_drugs))
        ]
    # Embedding coverage is required in both modes. (The historical code let
    # build_pair_features drop uncovered pairs after concatenation; filtering the
    # ID tables first is equivalent and avoids building the features at all.)
    pos_df = pos_df[pos_df["drug1"].isin(emb_drugs) & pos_df["drug2"].isin(emb_drugs)]
    neg_df = neg_df[neg_df["drug1"].isin(emb_drugs) & neg_df["drug2"].isin(emb_drugs)]

    print(f"\n[{level}] Eligible pairs (both drugs have a {level} embedding):")
    print(f"   Positives: {len(pos_df):,}")
    print(f"   Negatives: {len(neg_df):,}")

    # ----------------------------------------------------
    # 4. Balance on pair IDs, then build features for the chosen pairs only
    # ----------------------------------------------------
    if balance:
        n = min(len(pos_df), len(neg_df))
        print(f"   ⚖️  Balancing: {n} pos & {n} neg")
        df_bal = pd.concat([
            pos_df.sample(n, random_state=seed),
            neg_df.sample(n, random_state=seed),
        ], ignore_index=True)
        df_bal = df_bal.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        X_bal, y_bal, pairs_bal = build_pair_features(emb_df, df_bal)
        print(f"[{level}] FINAL dataset balanced: {len(pairs_bal):,} pairs")
        print(f"   → {int(sum(y_bal))} pos, {len(y_bal)-int(sum(y_bal))} neg")
        result = (X_bal, y_bal, pairs_bal)
        return (*result, pos_df.reset_index(drop=True).copy()) if return_positive_pool else result

    # ----------------------------------------------------
    # 5. If no balancing requested: every eligible pair (large!)
    # ----------------------------------------------------
    all_pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    print(f"   building features for all {len(all_pairs):,} eligible pairs")
    result = build_pair_features(emb_df, all_pairs)
    return (*result, pos_df.reset_index(drop=True).copy()) if return_positive_pool else result


def load_training_positive_pool(pos_path, neg_path, eligible_drugs):
    """Guarded eligible ID pool for non-main loaders (e.g. paired MeSH scopes)."""
    pos, _ = _resolve_supervision_labels(pd.read_csv(pos_path), pd.read_csv(neg_path), POLICY, None)
    ids = set(map(str, eligible_drugs))
    return pos[pos.drug1.isin(ids) & pos.drug2.isin(ids)].assign(label=1).reset_index(drop=True)


def create_folds(X, y, n_splits=5, seed=42, stratified=True):
    """Outer folds. Stratified by label (default), so every fold's test set is
    balanced like the cohort; ``stratified=False`` gives the historical shuffled
    KFold, which produced 1,045-1,102 test positives across the five folds."""
    if stratified:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(X, np.asarray(y)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))


def get_dataloaders(X, y, folds, fold_idx, batch_size=64):
    train_idx, test_idx = folds[fold_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = DDIDataset(X_train, y_train)
    test_ds = DDIDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_idx, test_idx

def load_inductive_test_pairs(inductive_drugs, pos_path, neg_path, emb_df,
                              balance=True, seed=42, conflict_policy=POLICY,
                              reference_pairs_path=None, scenario="either"):
    """
    Build DDI pairs for inductive evaluation:
    - Includes only pairs where at least one drug is inductive
    - Ensures both drugs have embedding vectors
    - Optionally balances positives and negatives 1:1
    Labels pass through the same R1.2 conflict guard as load_ddi_data.
    """
    import pandas as pd
    import numpy as np

    # Load files
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    pos_df, neg_df = _resolve_supervision_labels(
        pos_df, neg_df, conflict_policy, reference_pairs_path)
    pos_df["label"] = 1
    neg_df["label"] = 0

    emb_drugs = set(emb_df.index)

    # ----------------------------
    # Select inductive pairs only
    # ----------------------------
    if scenario not in ("either", "seen_unseen", "unseen_unseen"):
        raise ValueError("scenario must be either, seen_unseen, or unseen_unseen")
    def eligible(frame):
        a, b = frame.drug1.isin(inductive_drugs), frame.drug2.isin(inductive_drugs)
        mask = a | b if scenario == "either" else a ^ b if scenario == "seen_unseen" else a & b
        return frame[mask & frame.drug1.isin(emb_drugs) & frame.drug2.isin(emb_drugs)]
    # Eligibility precedes sampling: no class-dependent losses after balancing.
    pos_ind, neg_ind = eligible(pos_df), eligible(neg_df)

    print(f"[Inductive] Found {len(pos_ind)} positive and {len(neg_ind)} negative pairs before balancing.")

    # ----------------------------
    # Balance positives/negatives
    # ----------------------------
    if balance:
        n_neg = len(neg_ind)

        # Preserve a single-class cohort for explicitly limited reporting.
        if n_neg == 0 or len(pos_ind) == 0:
            print("[Inductive] Single-class test cohort; retaining available pairs.")
            all_ind = pd.concat([pos_ind, neg_ind], ignore_index=True)
        else:
            n_pos_target = min(len(pos_ind), n_neg)
            pos_sub = pos_ind.sample(n=n_pos_target, random_state=seed)
            neg_ind = neg_ind.sample(n=n_pos_target, random_state=seed)
            all_ind = (
                pd.concat([pos_sub, neg_ind], ignore_index=True)
                  .sample(frac=1.0, random_state=seed)
                  .reset_index(drop=True)
            )
            print(f"[Inductive] Balanced to {n_pos_target} positives and {n_pos_target} negatives.")
    else:
        # no balancing
        all_ind = pd.concat([pos_ind, neg_ind], ignore_index=True)
        print("[Inductive] Using unbalanced inductive test pairs.")

    # ----------------------------
    # Build pairwise input features
    # ----------------------------
    X_ind, y_ind, pairs_ind = build_pair_features(emb_df, all_ind)

    print(f"[Inductive] Final inductive test set → {len(pairs_ind)} pairs.")
    return X_ind, y_ind, pairs_ind

# ----------------------------------------------------------------------
# Inner validation split (K3 / independent review P1-3)
#
# Every training routine used the outer test fold for early stopping and
# checkpoint selection, so reported test metrics came from test-selected
# models. The helpers below carve a stratified validation subset out of each
# outer *training* fold; the outer test indices are never touched. Splitting
# happens before topology extraction so that both validation and test
# positive targets can be masked from the graph, and preprocessing is fitted
# on the inner-training rows only.
# ----------------------------------------------------------------------

def inner_split_seed(seed, fold_idx, repetition=None):
    """Seed of the validation assignment for one outer fold.

    Independent of the repetition unless `repetition` is given. A paired run
    (another graph arm) that uses the same seed, folds and val_frac therefore
    reproduces the identical assignment, which is what makes arm comparisons
    paired; the manifest written by `save_split_manifest` lets that be checked.
    """
    base = int(seed) + int(fold_idx)
    return base if repetition is None else base + 1000 * (int(repetition) + 1)


def create_inner_splits(y, folds, val_frac=0.15, seed=42, num_repetitions=1,
                        vary_by_repetition=False, repeat_outer=False):
    """Stratified validation subset inside every outer training fold.

    Returns ``splits[rep][fold] == (train_idx, val_idx, test_idx)`` as numpy
    arrays: `test_idx` is exactly the outer fold's test set, `val_idx` is drawn
    from the outer fold's training set only, and `train_idx` is the remainder.
    Validation is for early stopping / checkpoint selection; test is only for
    reporting.

    With ``vary_by_repetition=False`` (default) all repetitions share one
    assignment per fold, so the masked graph used for topology extraction is
    identical across repetitions and can be computed once per fold. With
    ``True`` each repetition gets its own assignment (seeded by
    `inner_split_seed(seed, fold, rep)`), and callers must extract topology per
    distinct mask -- `masked_pairs_key` gives the cache key for that.
    """
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    if repeat_outer:
        repeated = []
        for rep in range(num_repetitions):
            current = folds if rep == 0 else create_folds(
                np.zeros((len(y), 1)), y, n_splits=len(folds), seed=seed + 1000 * rep)
            repeated.append(create_inner_splits(
                y, current, val_frac=val_frac, seed=seed + 1000 * rep,
                num_repetitions=1, vary_by_repetition=False)[0])
        return repeated

    def one(fold_idx, rep):
        outer_train, outer_test = folds[fold_idx]
        outer_train = np.asarray(outer_train)
        outer_test = np.asarray(outer_test)
        tr, va = train_test_split(
            outer_train, test_size=val_frac,
            random_state=inner_split_seed(seed, fold_idx, rep),
            stratify=y[outer_train],
        )
        tr, va = np.sort(tr), np.sort(va)
        check_split_partition(tr, va, outer_test, outer_train)
        return tr, va, outer_test

    if vary_by_repetition:
        return [[one(f, rep) for f in range(len(folds))]
                for rep in range(num_repetitions)]
    shared = [one(f, None) for f in range(len(folds))]
    return [shared for _ in range(num_repetitions)]


def check_split_partition(train_idx, val_idx, test_idx, outer_train_idx=None):
    """Fatal checks: the three index sets are disjoint and, if the outer
    training set is given, train ∪ val is exactly that set."""
    tr, va, te = (set(map(int, a)) for a in (train_idx, val_idx, test_idx))
    if tr & va or tr & te or va & te:
        raise ValueError("train/val/test index sets overlap: "
                         f"{len(tr & va)} train∩val, {len(tr & te)} train∩test, "
                         f"{len(va & te)} val∩test")
    if not tr or not va or not te:
        raise ValueError("every partition must be non-empty "
                         f"(train={len(tr)}, val={len(va)}, test={len(te)})")
    if outer_train_idx is not None and tr | va != set(map(int, outer_train_idx)):
        raise ValueError("train ∪ val must equal the outer training fold")


def unordered_pair_keys(pairs_df):
    """Canonical (min, max) string identity of each row's drug pair."""
    return [tuple(sorted((str(a), str(b))))
            for a, b in zip(pairs_df["drug1"], pairs_df["drug2"])]


def masked_pairs_key(masked_pairs, drug_ids):
    """Hashable identity of a topology-extraction call: the positive unordered
    pairs removed from the graph plus the drugs featurised. Two (repetition,
    fold) runs with equal keys can share one extracted table; unequal keys
    must not, whatever their fold index."""
    pos = masked_pairs[masked_pairs["label"] == 1]
    return (frozenset(unordered_pair_keys(pos)), frozenset(map(str, drug_ids)))


def get_split_loaders(X, y, train_idx, val_idx, test_idx, batch_size=64):
    """Train/val/test loaders for the MeSH baseline with the StandardScaler
    fitted on the inner-training rows only (val and test are transformed)."""
    check_split_partition(train_idx, val_idx, test_idx)
    scaler = StandardScaler().fit(X[train_idx])

    def loader(idx, shuffle):
        ds = DDIDataset(scaler.transform(X[idx]), y[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loader(train_idx, True), loader(val_idx, False), loader(test_idx, False)


def split_manifest(splits, pairs, seed=None, val_frac=None, extra=None):
    """JSON-serialisable record of every (repetition, fold) assignment.

    Besides the index lists it stores the pair rows (drug ids + label) so an
    arm can verify it trained on the same assignment, and the number of
    unordered pair identities shared between the partitions (`overlap`).
    Overlaps are reported, not fatal: they come from duplicate rows in the
    supplied pair files (review finding P1-1) and are resolved upstream.
    """
    pairs = pairs.reset_index(drop=True)
    keys = unordered_pair_keys(pairs)
    manifest = {
        "seed": seed, "val_frac": val_frac,
        "n_pairs": int(len(pairs)),
        "num_repetitions": len(splits),
        "num_folds": len(splits[0]) if splits else 0,
        "shared_across_repetitions": all(s is splits[0] for s in splits),
        "repetitions": [],
    }
    if extra:
        manifest.update(extra)
    for rep, per_fold in enumerate(splits):
        rep_entry = []
        for fold_idx, (tr, va, te) in enumerate(per_fold):
            part_keys = {name: {keys[i] for i in idx}
                         for name, idx in (("train", tr), ("val", va), ("test", te))}
            overlap = {
                "train_val": len(part_keys["train"] & part_keys["val"]),
                "train_test": len(part_keys["train"] & part_keys["test"]),
                "val_test": len(part_keys["val"] & part_keys["test"]),
            }
            if any(overlap.values()):
                print(f"WARNING: repetition {rep} fold {fold_idx}: unordered pair "
                      f"identities shared between partitions {overlap}")
            rep_entry.append({
                "fold": fold_idx,
                "train_idx": [int(i) for i in tr],
                "val_idx": [int(i) for i in va],
                "test_idx": [int(i) for i in te],
                "n": {"train": len(tr), "val": len(va), "test": len(te)},
                "n_pos": {name: int(pairs.loc[idx, "label"].sum())
                          for name, idx in (("train", tr), ("val", va), ("test", te))},
                "overlap_unordered_pairs": overlap,
            })
        manifest["repetitions"].append(rep_entry)
    manifest["pairs"] = [
        {"drug1": str(a), "drug2": str(b), "label": int(l)}
        for a, b, l in zip(pairs["drug1"], pairs["drug2"], pairs["label"])
    ]
    return manifest


def save_split_manifest(splits, pairs, path, **kwargs):
    import json
    manifest = split_manifest(splits, pairs, **kwargs)
    with open(path, "w") as f:
        json.dump(manifest, f)
    print(f"Saved split manifest → {path}")
    return manifest


def load_split_manifest(path, pairs=None):
    """Read a manifest back into ``splits[rep][fold]`` tuples. If `pairs` is
    given, assert the manifest was produced from the same pair rows."""
    import json
    with open(path) as f:
        manifest = json.load(f)
    if pairs is not None:
        pairs = pairs.reset_index(drop=True)
        recorded = [(p["drug1"], p["drug2"], p["label"]) for p in manifest["pairs"]]
        current = [(str(a), str(b), int(l))
                   for a, b, l in zip(pairs["drug1"], pairs["drug2"], pairs["label"])]
        if recorded != current:
            raise ValueError(f"{path} was produced from different pair rows "
                             "than the ones supplied; do not reuse its assignment")
    splits = [[(np.asarray(e["train_idx"]), np.asarray(e["val_idx"]),
                np.asarray(e["test_idx"])) for e in rep]
              for rep in manifest["repetitions"]]
    for rep in splits:
        for tr, va, te in rep:
            check_split_partition(tr, va, te)
    return splits, manifest
