import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm



PAIR_BINS = [
    ("low_level", "low_level"),
    ("low_level", "mid_level"),
    ("low_level", "deep_level"),
    ("mid_level", "mid_level"),
    ("mid_level", "deep_level"),
    ("deep_level", "deep_level"),
]
PAIR_BIN_NAMES = {
    ("low_level", "low_level"): "low-low",
    ("low_level", "mid_level"): "low-mid",
    ("low_level", "deep_level"): "low-deep",
    ("mid_level", "mid_level"): "mid-mid",
    ("mid_level", "deep_level"): "mid-deep",
    ("deep_level", "deep_level"): "deep-deep",
}

def _canon_pair(c1, c2):
    """Return canonical unordered pair (min, max) according to order low<mid<deep."""
    order = {"low_level": 0, "mid_level": 1, "deep_level": 2}
    if pd.isna(c1) or pd.isna(c2):
        return None
    a, b = sorted([c1, c2], key=lambda c: order.get(c, -1))
    key = (a, b)
    return key if key in PAIR_BIN_NAMES else None


def train_one_fold(
    model,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    patience=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the model for one CV fold with early stopping based on validation AUC.
    Returns the best-performing model and training history.
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    train_dataset_len = len(train_loader.dataset)
    val_dataset_len = len(val_loader.dataset)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, n_epochs + 1):
        # === Training ===
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)

        train_loss = running_loss / train_dataset_len
        history["train_loss"].append(train_loss)

        # === Validation ===
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * len(yb)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_loss /= val_dataset_len
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        auc = roc_auc_score(val_labels, val_preds)

        history["val_loss"].append(val_loss)
        history["val_auc"].append(auc)

        print(f"Epoch {epoch:02d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc:.4f}")

        # === Early Stopping Logic ===
        if auc > best_auc + 1e-4:  # improvement threshold
            best_auc = auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏸Early stopping triggered (no improvement for {patience} epochs).")
            break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("No improvement observed during training — returning last model state.")

    print(f"Best Validation AUC: {best_auc:.4f}")
    return model, history

def evaluate_by_pair_bins(model, loader, pairs_df, drug_to_cat,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          min_samples=30):
    """
    Evaluate metrics for each unordered pair bin:
      low-low, low-mid, low-deep, mid-mid, mid-deep, deep-deep.
    Skips bins with < min_samples to avoid unstable metrics.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.cpu().numpy())
            labels.extend(yb.numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    binary = (preds >= 0.5).astype(int)

    df = pairs_df.copy().reset_index(drop=True)
    df["pred"] = preds
    df["binary"] = binary
    df["cat1"] = df["drug1"].map(drug_to_cat)
    df["cat2"] = df["drug2"].map(drug_to_cat)
    df["pair_bin"] = df.apply(lambda r: _canon_pair(r["cat1"], r["cat2"]), axis=1)
    df["pair_bin_name"] = df["pair_bin"].map(PAIR_BIN_NAMES)

    results = {}
    for key in PAIR_BINS:
        name = PAIR_BIN_NAMES[key]
        sub = df[df["pair_bin"] == key]
        n = len(sub)
        if n < min_samples:
            # skip tiny bins; you can log if you want
            continue
        y_true = sub["label"].values
        y_prob = sub["pred"].values
        y_hat = sub["binary"].values

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan  # all-one/all-zero label edge-cases

        results[name] = {
            "n_samples": n,
            "accuracy": accuracy_score(y_true, y_hat),
            "precision": precision_score(y_true, y_hat, zero_division=0),
            "recall": recall_score(y_true, y_hat, zero_division=0),
            "f1": f1_score(y_true, y_hat, zero_division=0),
            "auc": auc,
            "confusion_matrix": confusion_matrix(y_true, y_hat),
        }
    return results