"""Validation-AUROC early stopping with restoration of the selected checkpoint."""
import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam


def train_one_fold(
    model,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    patience=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epoch_loader=None,
    verbose=True,
):
    """
    Train the model for one CV fold with early stopping based on validation AUC.
    Returns the best-performing model and training history.
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_auc = -np.inf
    best_epoch = None
    best_state = None
    patience_counter = 0

    train_dataset_len = len(train_loader.dataset)
    val_dataset_len = len(val_loader.dataset)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, n_epochs + 1):
        # === Training ===
        model.train()
        running_loss = 0.0
        current_loader = train_loader if epoch_loader is None else epoch_loader(epoch - 1)
        for xb, yb in current_loader:
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

        if verbose:
            print(f"Epoch {epoch:02d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc:.4f}")

        # === Early Stopping Logic ===
        if auc > best_auc + 1e-4:  # improvement threshold
            best_auc = auc
            best_epoch = epoch
            # state_dict() returns live references; without a copy the
            # 'best' snapshot silently tracks later updates and the restored
            # model is the LAST epoch, not the selected one.
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"⏸Early stopping triggered (no improvement for {patience} epochs).")
            break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("No improvement observed during training — returning last model state.")

    # Argmax can name a later epoch whose gain was smaller than min_delta.
    # Record the checkpoint actually restored, rather than that later maximum.
    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_auc
    if verbose:
        print(f"Best Validation AUC: {best_auc:.4f}")
    return model, history
