import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch


class DDIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_pair_features(emb_df, pairs_df):
    """Create pairwise input features by concatenating drug embeddings."""
    emb_dict = {drug: row.values for drug, row in emb_df.iterrows()}
    X, y, valid_pairs = [], [], []

    missing = 0
    for _, row in pairs_df.iterrows():
        d1, d2, label = row["drug1"], row["drug2"], row["label"]
        if d1 in emb_dict and d2 in emb_dict:
            feat = np.concatenate([emb_dict[d1], emb_dict[d2]])
            X.append(feat)
            y.append(label)
            valid_pairs.append((d1, d2, label))
        else:
            missing += 1

    if missing > 0:
        print(f"⚠️  Skipped {missing} pairs with missing embeddings.")

    X = np.array(X)
    y = np.array(y)
    pairs = pd.DataFrame(valid_pairs, columns=["drug1", "drug2", "label"])
    return X, y, pairs


def load_ddi_data(level: str, pos_path: str, neg_path: str, emb_dir: str, seed=42, max_diff=500):
    """
    Load embeddings + DDI pairs for a given knowledge level.
    Balances positives vs negatives adaptively and keeps pair ordering.
    Returns: X, y, pairs_df (aligned row-by-row)
    """
    np.random.seed(seed)

    # --- Load embeddings ---
    emb_path = f"{emb_dir}/MeSH_{level}_tfidf_svd128.csv"
    emb_df = pd.read_csv(emb_path, index_col=0)
    emb_drugs = set(emb_df.index)

    # --- Load raw pairs ---
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    pos_df["label"], neg_df["label"] = 1, 0

    # --- Keep only pairs with both drugs in embedding space ---
    pos_df = pos_df[pos_df["drug1"].isin(emb_drugs) & pos_df["drug2"].isin(emb_drugs)]
    neg_df = neg_df[neg_df["drug1"].isin(emb_drugs) & neg_df["drug2"].isin(emb_drugs)]

    n_pos_all, n_neg_all = len(pos_df), len(neg_df)
    print(f"\n[{level}] Drug coverage check:")
    print(f"   Positives covered: {n_pos_all:,}")
    print(f"   Negatives covered: {n_neg_all:,}")

    if n_pos_all == 0 or n_neg_all == 0:
        raise ValueError(f"[{level}] No positive or negative pairs found with embeddings!")

    # --- Adaptive sampling of positives to roughly match negatives ---
    n_target_neg = n_neg_all
    diff = float("inf")
    sample_size = min(n_pos_all, n_target_neg)
    attempt = 0

    while diff > max_diff and sample_size > 0:
        pos_sample = pos_df.sample(n=sample_size, random_state=seed + attempt)
        n_pos_sample = len(pos_sample)
        diff = abs(n_target_neg - n_pos_sample)
        attempt += 1
        if diff > max_diff:
            sample_size = max(0, sample_size - 250)

    print(f"   ⚖️  Sampled {n_pos_sample:,} positives to match {n_target_neg:,} negatives (Δ={diff}).")

    # --- Merge + shuffle together ---
    all_pairs = pd.concat([pos_sample, neg_df], ignore_index=True)
    all_pairs = all_pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- Build features and ensure alignment ---
    X, y, pairs = build_pair_features(emb_df, all_pairs)
    print(f"[{level}] Final dataset: {len(pairs):,} pairs ({int(sum(y))} pos, {len(y)-int(sum(y))} neg)")
    print(f"→ Feature shape: {X.shape}")

    return X, y, pairs


def create_folds(X, y, n_splits=5, seed=42):
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