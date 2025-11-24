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
        print(f"Skipped {missing} pairs with missing embeddings.")

    X = np.array(X)
    y = np.array(y)
    pairs = pd.DataFrame(valid_pairs, columns=["drug1", "drug2", "label"])
    return X, y, pairs


def load_ddi_data(level: str, pos_path: str, neg_path: str, emb_dir: str,
                  seed=42, max_diff=500, inductive_drugs=None,
                  balance=True):

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
    else:
        pos_df = pos_df[
            pos_df["drug1"].isin(emb_drugs) &
            pos_df["drug2"].isin(emb_drugs)
        ]
        neg_df = neg_df[
            neg_df["drug1"].isin(emb_drugs) &
            neg_df["drug2"].isin(emb_drugs)
        ]

    print(f"\n[{level}] Raw coverage:")
    print(f"   Positives: {len(pos_df):,}")
    print(f"   Negatives: {len(neg_df):,}")

    # ----------------------------------------------------
    # 4. Pre-merge BEFORE embedding filtering
    # ----------------------------------------------------
    all_pairs_raw = pd.concat([pos_df, neg_df], ignore_index=True)

    # ----------------------------------------------------
    # 5. Build features (drops missing embedding pairs)
    # ----------------------------------------------------
    X_all, y_all, pairs_all = build_pair_features(emb_df, all_pairs_raw)

    print(f"[{level}] After embedding filtering:")
    print(f"   Total pairs: {len(pairs_all):,}")
    print(f"   Positives: {sum(y_all):,}")
    print(f"   Negatives: {len(y_all) - sum(y_all):,}")

    # ----------------------------------------------------
    # 6. FINAL BALANCING (AFTER filtering)
    # ----------------------------------------------------
    if balance:
        df = pairs_all.copy()
        pos = df[df.label == 1]
        neg = df[df.label == 0]

        n = min(len(pos), len(neg))

        print(f"   ⚖️  Balancing AFTER filtering: {n} pos & {n} neg")

        df_bal = pd.concat([
            pos.sample(n, random_state=seed),
            neg.sample(n, random_state=seed)
        ], ignore_index=True)

        df_bal = df_bal.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        X_bal, y_bal, pairs_bal = build_pair_features(emb_df, df_bal)

        print(f"[{level}] FINAL dataset balanced: {len(pairs_bal):,} pairs")
        print(f"   → {sum(y_bal)} pos, {len(y_bal)-sum(y_bal)} neg")

        return X_bal, y_bal, pairs_bal

    # ----------------------------------------------------
    # 7. If no balancing requested
    # ----------------------------------------------------
    return X_all, y_all, pairs_all


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

def load_inductive_test_pairs(inductive_drugs, pos_path, neg_path, emb_df,
                              balance=True, seed=42):
    """
    Build DDI pairs for inductive evaluation:
    - Includes only pairs where at least one drug is inductive
    - Ensures both drugs have embedding vectors
    - Optionally balances positives and negatives 1:1
    """
    import pandas as pd
    import numpy as np

    # Load files
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    pos_df["label"] = 1
    neg_df["label"] = 0

    emb_drugs = set(emb_df.index)

    # ----------------------------
    # Select inductive pairs only
    # ----------------------------
    pos_ind = pos_df[
        ((pos_df["drug1"].isin(inductive_drugs)) |
         (pos_df["drug2"].isin(inductive_drugs)))
        & (pos_df["drug1"].isin(emb_drugs))
        & (pos_df["drug2"].isin(emb_drugs))
    ]

    neg_ind = neg_df[
        ((neg_df["drug1"].isin(inductive_drugs)) |
         (neg_df["drug2"].isin(inductive_drugs)))
        & (neg_df["drug1"].isin(emb_drugs))
        & (neg_df["drug2"].isin(emb_drugs))
    ]

    print(f"[Inductive] Found {len(pos_ind)} positive and {len(neg_ind)} negative pairs before balancing.")

    # ----------------------------
    # Balance positives/negatives
    # ----------------------------
    if balance:
        n_neg = len(neg_ind)

        # If there are no negatives, do not sample
        if n_neg == 0:
            print("[Inductive] WARNING: No negative inductive pairs found! Returning positives only.")
            all_ind = pos_ind.copy()
        else:
            n_pos_target = min(len(pos_ind), n_neg)
            pos_sub = pos_ind.sample(n=n_pos_target, random_state=seed)
            all_ind = (
                pd.concat([pos_sub, neg_ind], ignore_index=True)
                  .sample(frac=1.0, random_state=seed)
                  .reset_index(drop=True)
            )
            print(f"[Inductive] Balanced to {n_pos_target} positives and {n_neg} negatives.")
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