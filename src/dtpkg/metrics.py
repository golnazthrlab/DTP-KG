"""Shared pair-category metrics for probability scores."""
import numpy as np
import pandas as pd
from dtpkg.evaluation_inputs import pair_drugs, validate_feature_coverage
from sklearn.metrics import (average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)


LEVEL_ORDER = {"low_level": 0, "mid_level": 1, "deep_level": 2}

def _canon_pair(c1, c2):
    """Canonicalize category pairs (order-independent, low<mid<deep)."""
    ordered = sorted([c1, c2], key=lambda c: LEVEL_ORDER.get(c, 99))
    return "-".join(ordered)

PAIR_BINS = [
    "low_level-low_level",
    "low_level-mid_level",
    "low_level-deep_level",
    "mid_level-mid_level",
    "mid_level-deep_level",
    "deep_level-deep_level",
]

def _bin_result_dict(y_true, y_prob, y_hat):
    if len(y_true) == 0:
        return {**dict.fromkeys(("n", "n_pos", "n_neg", "tp", "fp", "tn", "fn"), 0),
                **dict.fromkeys(("acc", "prec", "rec", "f1", "auc", "ap", "specificity"), np.nan)}
    # confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    # auc can fail on single-class; guard
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    return {
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "acc": accuracy_score(y_true, y_hat),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "prec": precision_score(y_true, y_hat, zero_division=0),
        "rec": recall_score(y_true, y_hat, zero_division=0),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "auc": auc,
        "ap": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
    }

def evaluate_pair_scores(pairs, scores, drug_to_cat, threshold=.5):
    """Shared evaluator for raw graph scores and neural classifiers."""
    df = pairs.copy().reset_index(drop=True)
    if len(df) != len(scores):
        raise ValueError("one score per evaluation pair is required")
    df["pred"] = np.asarray(scores, float)
    df["threshold"] = float(threshold)
    df["binary"] = (df.pred >= threshold).astype(int)
    df["cat1"] = df.drug1.map(drug_to_cat)
    df["cat2"] = df.drug2.map(drug_to_cat)
    df["pair_bin"] = [_canon_pair(a,b) for a,b in zip(df.cat1,df.cat2)]
    results = {}
    for key in PAIR_BINS + ["overall"]:
        g = df if key == "overall" else df[df.pair_bin == key]
        results[key] = _bin_result_dict(g.label.to_numpy(int), g.pred.to_numpy(), g.binary.to_numpy())
    return results, df


def evaluate_by_pair_bins(model, loader, pairs_df, drug_to_cat,
                          device=None, min_samples=0, return_predictions=False):
    """
    Baseline eval (MeSH-only). Returns dict[bin_name] -> metric dict.
    min_samples=0 so we don't silently drop bins.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            preds.extend(probs)
            labels.extend(yb.numpy())

    preds = np.asarray(preds)
    labels = np.asarray(labels).astype(int)
    binary = (preds >= 0.5).astype(int)

    df = pairs_df.copy().reset_index(drop=True)
    df["pred"] = preds
    df["binary"] = binary
    df["cat1"] = df["drug1"].map(drug_to_cat)
    df["cat2"] = df["drug2"].map(drug_to_cat)
    df["pair_bin"] = df.apply(lambda r: _canon_pair(r["cat1"], r["cat2"]), axis=1)

    results = {}
    for key in PAIR_BINS + ["overall"]:
        sub = df if key == "overall" else df[df["pair_bin"] == key]
        if len(sub) < min_samples:
            continue
        y_true = sub["label"].astype(int).to_numpy()
        y_prob = sub["pred"].to_numpy()
        y_hat  = sub["binary"].astype(int).to_numpy()
        results[key] = _bin_result_dict(y_true, y_prob, y_hat)
    return (results, df) if return_predictions else results


def evaluate_by_pair_bins_fusion(model, pairs_df, emb_df, topo_df, drug_to_cat,
                                 device=None, min_samples=0, return_predictions=False):
    """
    Fusion eval (needs per-drug bio+topo lookups).
    Returns dict[bin_name] -> metric dict.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    validate_feature_coverage(emb_df, pair_drugs(pairs_df), "MeSH")
    validate_feature_coverage(topo_df, pair_drugs(pairs_df), "topology")
    model.eval()

    emb_dict  = emb_df.to_dict(orient="index")
    topo_dict = topo_df.to_dict(orient="index")
    emb_cols  = list(emb_df.columns)
    topo_cols = list(topo_df.columns)
    bio_dim   = emb_df.shape[1]
    topo_dim  = topo_df.shape[1]

    def get_feat(drug, src, cols, dim):
        if drug not in src:
            raise ValueError(f"Missing features for drug {drug!r}; zero imputation is disabled")
        return torch.tensor([src[drug][c] for c in cols], dtype=torch.float32)

    preds, labels = [], []

    with torch.no_grad():
        for _, row in pairs_df.iterrows():
            d1, d2, label = row["drug1"], row["drug2"], int(row["label"])
            bio1  = get_feat(d1, emb_dict,  emb_cols,  bio_dim).unsqueeze(0).to(device)
            topo1 = get_feat(d1, topo_dict, topo_cols, topo_dim).unsqueeze(0).to(device)
            bio2  = get_feat(d2, emb_dict,  emb_cols,  bio_dim).unsqueeze(0).to(device)
            topo2 = get_feat(d2, topo_dict, topo_cols, topo_dim).unsqueeze(0).to(device)

            logits, _ = model(bio1, topo1, bio2, topo2)
            prob = torch.sigmoid(logits).item()
            preds.append(prob)
            labels.append(label)

    preds  = np.asarray(preds)
    labels = np.asarray(labels).astype(int)
    binary = (preds >= 0.5).astype(int)

    df = pairs_df.copy().reset_index(drop=True)
    df["pred"] = preds
    df["binary"] = binary
    df["cat1"] = df["drug1"].map(drug_to_cat)
    df["cat2"] = df["drug2"].map(drug_to_cat)
    df["pair_bin"] = df.apply(lambda r: _canon_pair(r["cat1"], r["cat2"]), axis=1)

    results = {}
    for key in PAIR_BINS + ["overall"]:
        sub = df if key == "overall" else df[df["pair_bin"] == key]
        if len(sub) < min_samples:
            continue
        y_true = sub["label"].astype(int).to_numpy()
        y_prob = sub["pred"].to_numpy()
        y_hat  = sub["binary"].astype(int).to_numpy()
        results[key] = _bin_result_dict(y_true, y_prob, y_hat)
    return (results, df) if return_predictions else results


def evaluate_baseline_inductive(model, pairs_df, emb_df, drug_to_cat, device, scaler=None, return_predictions=False):
    model.eval()
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    preds, labels = [], []

    # ------------------------------
    # Build safe embedding dictionary
    # ------------------------------
    emb_dict = emb_df.to_dict(orient="index")
    cols = list(emb_df.columns)

    def safe_vec(value, drug, col):
        """Ensure the embedding value is a 1D numpy array."""
        v = np.asarray(value)
        if v.ndim == 0:
            # scalar → wrap into 1D
            return v.reshape(1)
        if v.ndim == 1:
            return v
        raise ValueError(f"Unexpected dim for {drug}:{col} → shape {v.shape}")

    with torch.no_grad():
        for _, row in pairs_df.iterrows():
            d1, d2, y = row["drug1"], row["drug2"], row["label"]

            # ---- check existence ----
            if d1 not in emb_dict:
                raise KeyError(f"Drug {d1} missing in emb_df index.")
            if d2 not in emb_dict:
                raise KeyError(f"Drug {d2} missing in emb_df index.")

            # ---- build feature vector ----
            vec1 = [safe_vec(emb_dict[d1][c], d1, c) for c in cols]
            vec2 = [safe_vec(emb_dict[d2][c], d2, c) for c in cols]
            a, b = np.concatenate(vec1), np.concatenate(vec2)
            x = np.concatenate([a + b, np.abs(a - b)])
            if scaler is not None:
                x = scaler.transform(x.reshape(1, -1)).ravel()

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().item()
            preds.append(prob)
            labels.append(y)

    df = pairs_df.copy()
    df["pred"] = preds
    df["binary"] = (df["pred"] >= 0.5).astype(int)
    df["cat1"] = df["drug1"].map(drug_to_cat)
    df["cat2"] = df["drug2"].map(drug_to_cat)
    df["pair_bin"] = df.apply(lambda r: _canon_pair(r["cat1"], r["cat2"]), axis=1)

    # Keep empty bins with counts and undefined metrics, like the fusion evaluator.
    out = {}
    for key in PAIR_BINS + ["overall"]:
        sub = df if key == "overall" else df[df["pair_bin"] == key]
        y_true = sub["label"].astype(int).to_numpy()
        y_prob = sub["pred"].to_numpy()
        y_hat  = sub["binary"].astype(int).to_numpy()
        out[key] = _bin_result_dict(y_true, y_prob, y_hat)

    return (out, df) if return_predictions else out
