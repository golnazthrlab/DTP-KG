"""Topology-only comparator used in the transductive fusion experiment."""
import numpy as np
import torch
from torch import nn, optim

from dtpkg.evaluation_inputs import pair_drugs, validate_feature_coverage
from dtpkg.fusion.trainer import EarlyStopping

class TopoOnlyModel(nn.Module):
    def __init__(self, topo_dim, hidden=(64, 32), dropout=0.1, pair_mode="sym"):
        super().__init__()
        if pair_mode not in ("sym", "concat"):
            raise ValueError("pair_mode must be 'sym' or 'concat'")
        self.pair_mode = pair_mode

        # Total topo dimension after concatenating topo1 & topo2
        in_dim = 2 * topo_dim

        # Add LayerNorm BEFORE MLP
        self.input_norm = nn.LayerNorm(in_dim)
        layers = []
        d = in_dim

        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, topo1, topo2, drop_topo1=0.0, drop_topo2=0.0):
        # drop_topo1/drop_topo2 are accepted for API compatibility with
        # FusionTrainer._evaluate_bins_topo_only (which passes modality-dropout
        # probs); a topo-only model has no modality to drop, so they are no-ops.
        x = torch.cat([topo1 + topo2, torch.abs(topo1 - topo2)]
                      if self.pair_mode == "sym" else [topo1, topo2], dim=1)
        # x = self.input_norm(x)   # normalize before MLP
        return self.net(x).view(-1)



class TopologyBaseline:
    """Fit the topology comparator with the fusion trainer's split and sampling policy."""

    def __init__(self, trainer):
        self.trainer = trainer

    def _train_topo_only(
        self,
        model,
        train_pairs,
        val_pairs,
        topo_df,
        *,
        test_pairs=None,
        patience=None,
        max_epochs=None,
        lr=None,
        weight_decay=None,
        device=None,
        epoch_sampler=None,
        fit_seed=None,
        normalization_drugs=None,
    ):
        """Fit on `train_pairs`, early-stop / select on `val_pairs`, score
        `test_pairs` once with the restored best model (None -> no scoring;
        keyword-only so a legacy positional test fold cannot become the
        stopping set). Normalisation statistics come from the drugs incident
        to the inner-training pairs only."""
        # None -> the limits declared on the trainer (one source of truth)
        max_epochs = self.trainer.epochs if max_epochs is None else max_epochs
        lr = self.trainer.lr if lr is None else lr
        weight_decay = self.trainer.weight_decay if weight_decay is None else weight_decay
        device = self.trainer.device if device is None else device
        required_drugs = self.trainer.sampler_drugs(epoch_sampler, train_pairs) | pair_drugs(val_pairs)
        if test_pairs is not None:
            required_drugs |= pair_drugs(test_pairs)
        validate_feature_coverage(topo_df, required_drugs, "topology")
        stopper = EarlyStopping(
            patience=self.trainer.patience if patience is None else patience,
            min_delta=self.trainer.min_delta,
        )
        model = self.trainer._fresh_model(model, fit_seed).to(device)
        self.trainer.last_predictions = self.trainer.last_gates = None
        self.trainer.last_epoch_losses = []
        optim_ = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # ---------------------------
        # Build lookup dict
        # ---------------------------
        topo_cols = list(topo_df.columns)
        topo_dict = topo_df.to_dict(orient="index")

        # ---------------------------
        # Compute train-only normalization stats
        # ---------------------------
        scaling_pairs = train_pairs if epoch_sampler is None else epoch_sampler.epoch(0)
        train_drugs = set(scaling_pairs["drug1"]) | set(scaling_pairs["drug2"])
        if normalization_drugs is not None:
            allowed = self.trainer.sampler_drugs(epoch_sampler, train_pairs)
            train_drugs = set(normalization_drugs)
            if not train_drugs or not train_drugs.issubset(allowed):
                raise ValueError("topology scaling drugs must belong to the permitted training population")
        train_mat = topo_df.loc[sorted(train_drugs)].values

        mu = train_mat.mean(axis=0)
        sigma = train_mat.std(axis=0) + 1e-6  # avoid division by zero

        mu_t = torch.tensor(mu, dtype=torch.float32).to(device)
        sigma_t = torch.tensor(sigma, dtype=torch.float32).to(device)

        # ---------------------------
        # Helper to retrieve normalized topo feature vector
        # ---------------------------
        def get_norm_topo(drug):
            raw = torch.tensor(
                [topo_dict[str(drug)][c] for c in topo_cols],
                dtype=torch.float32,
                device=device,
            )
            return (raw - mu_t) / sigma_t

        # ---------------------------
        # Training loop
        # ---------------------------
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            train_losses = []
            train_by_class = {0: [], 1: []}

            epoch_pairs = train_pairs if epoch_sampler is None else epoch_sampler.epoch(epoch)
            for _, row in epoch_pairs.iterrows():
                d1, d2, y = row["drug1"], row["drug2"], row["label"]

                t1 = get_norm_topo(d1).unsqueeze(0)
                t2 = get_norm_topo(d2).unsqueeze(0)
                yb = torch.tensor([y], dtype=torch.float32, device=device)

                optim_.zero_grad()
                logits = model(t1, t2)
                loss = criterion(logits, yb)
                loss.backward()
                optim_.step()

                train_losses.append(loss.item())
                train_by_class[int(y)].append(loss.item())

            # ---------------------------
            # Validation (inner split; the outer test fold is never seen here)
            # ---------------------------
            model.eval()
            val_losses = []
            val_by_class = {0: [], 1: []}
            with torch.no_grad():
                for _, row in val_pairs.iterrows():
                    d1, d2, y = row["drug1"], row["drug2"], row["label"]

                    t1 = get_norm_topo(d1).unsqueeze(0)
                    t2 = get_norm_topo(d2).unsqueeze(0)
                    yb = torch.tensor([y], dtype=torch.float32, device=device)

                    logits = model(t1, t2)
                    loss_value = criterion(logits, yb).item()
                    val_losses.append(loss_value)
                    val_by_class[int(y)].append(loss_value)

            self.trainer.last_epoch_losses.append(self.trainer._class_loss_row(epoch, train_by_class, val_by_class))
            mean_val_loss = np.mean(val_losses)

            # Early stopping logic
            if stopper.step(mean_val_loss):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if stopper.should_stop:
                print(f"Early stop at epoch {epoch+1} "
                      f"(best val loss {stopper.best:.4f} at epoch {stopper.best_epoch+1})")
                break

        # Load best state
        model.load_state_dict(best_state)

        self.trainer.last_fit_info = {
            "fit_seed": self.trainer.seed if fit_seed is None else fit_seed,
            "epochs_run": epoch + 1,
            "best_epoch": stopper.best_epoch + 1,
            "best_val_loss": float(stopper.best),
            "n_train": len(train_pairs), "n_val": len(val_pairs),
            "n_test": None if test_pairs is None else len(test_pairs),
            "positive_sampling": "fixed" if epoch_sampler is None else epoch_sampler.mode,
        }
        if epoch_sampler is not None:
            self.trainer.last_fit_info.update(epoch_sampler.describe(epoch + 1))
            self.trainer.last_fit_info["n_train"] = epoch_sampler.epoch_size
            self.trainer.last_fit_info["unique_training_positives"] = epoch_sampler.unique_positives(epoch + 1)
        self.last_model = model
        self.last_normalization = (mu, sigma, topo_cols)
        if test_pairs is None:
            return None

        # ---------------------------
        # Outer test fold: scored once, with the validation-selected model
        # ---------------------------
        eval_bins = self.trainer._evaluate_bins_topo_only(
            model, topo_df, test_pairs, mu, sigma, topo_cols
        )

        return eval_bins
