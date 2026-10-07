"""Cross-validation and inductive fitting with trusted local checkpoint replay."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dtpkg.ddi_sampling import EpochPositiveSampler, epoch_seed
from dtpkg.evaluation_inputs import pair_drugs, validate_drug_depths, validate_feature_coverage
from dtpkg.data_loaders import (
    DDIDataset, get_split_loaders, create_inner_splits, save_split_manifest,
    masked_pairs_key, build_pair_features,
)
from dtpkg.metrics import evaluate_by_pair_bins, evaluate_by_pair_bins_fusion, evaluate_baseline_inductive


class EarlyStopping:
    """Explicit improvement/patience counter on a minimised validation metric.

    `step(loss)` returns True when `loss` improves on the best seen so far by
    more than `min_delta` (and resets the counter); otherwise the counter is
    incremented. `should_stop` is True once `patience` consecutive epochs have
    passed without improvement.
    """

    def __init__(self, patience=10, min_delta=1e-4):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.wait = 0
        self.epoch = -1

    def step(self, loss):
        self.epoch += 1
        if loss < self.best - self.min_delta:
            self.best = loss
            self.best_epoch = self.epoch
            self.wait = 0
            return True
        self.wait += 1
        return False

    @property
    def should_stop(self):
        return self.wait >= self.patience


class FusionTrainer:
    def __init__(
        self,
        fusion_model_wo_go=None,
        fusion_model_go=None,
        baseline_cls=None,
        topo_extractor=None,
        topo_extractor_with_go=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_splits=5,
        batch_size=128,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-5,
        seed=42,
        patience=10,
        min_delta=1e-4,
        out_dir="results",
        drop_bio_prob=0,
        drop_topo_prob=0.0,
        val_frac=0.15,
        inner_val_per_repetition=False,
        positive_sampling="per_epoch",
        positive_to_negative_ratio=1.0,
        positive_pool=None,
        repeat_outer_splits=False,
        include_topology_baseline=False,
        include_graph_baselines=False,
    ):
        self.fusion_model_wo_go = fusion_model_wo_go
        self.fusion_model_go = fusion_model_go
        self.baseline_cls = baseline_cls
        self.topo_extractor = topo_extractor
        self.topo_extractor_with_go = topo_extractor_with_go
        self.device = device
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        if positive_sampling not in ("per_epoch", "fixed"):
            raise ValueError("positive_sampling must be 'per_epoch' or 'fixed'")
        if not np.isfinite(positive_to_negative_ratio) or positive_to_negative_ratio <= 0:
            raise ValueError("positive_to_negative_ratio must be finite and > 0")
        self.positive_sampling = positive_sampling
        self.positive_to_negative_ratio = positive_to_negative_ratio
        # Full eligible positive ID table, NOT the small balanced evaluation cohort.
        self.positive_pool = positive_pool
        self.repeat_outer_splits = repeat_outer_splits
        self.include_topology_baseline = include_topology_baseline
        self.include_graph_baselines = include_graph_baselines
        self.last_epoch_losses = []
        self.last_predictions = None
        self.last_gates = None
        self.patience = patience
        self.min_delta = min_delta
        self.out_dir = out_dir
        self.drop_bio_prob = drop_bio_prob
        self.drop_topo_prob = drop_topo_prob
        # Inner validation split (K3): a stratified `val_frac` of every outer
        # training fold is held out for early stopping and checkpoint selection;
        # the outer test fold is only ever scored once, with the restored best
        # model. With `inner_val_per_repetition=False` every repetition shares
        # one assignment per fold (topology extracted once per fold); with True
        # each repetition draws its own and topology is extracted per distinct
        # mask. Either way the assignment is a function of (seed, fold[, rep]),
        # so paired graph arms run with the same seed/folds share it.
        self.val_frac = val_frac
        self.inner_val_per_repetition = inner_val_per_repetition
        self.splits = None
        self.training_log = []
        self.last_fit_info = None
        os.makedirs(out_dir, exist_ok=True)

        # --- Initialize result containers dynamically ---
        self.results = {"baseline": [], "fusion": []}
        # Per-fold extractor diagnostics (backend, Katz status, closeness shortcut,
        # per-drug seconds), so a result can state what was actually computed.
        self.topo_diagnostics = {}
        if self.topo_extractor_with_go is not None and self.fusion_model_go is not None:
            self.results["fusion_with_go"] = []

        print("FusionTrainer initialized.")
        print(f"   → Using GO extractor: {self.topo_extractor_with_go is not None}")


    @staticmethod
    def _class_loss_row(epoch, training, validation):
        row = {"epoch": epoch}
        for phase, values in (("train", training), ("validation", validation)):
            for label in (0, 1):
                row[f"{phase}_n_{label}"] = len(values[label])
                row[f"{phase}_bce_{label}"] = float(np.mean(values[label])) if values[label] else np.nan
        return row


    def _fresh_model(self, template, fit_seed=None):
        """Fresh seeded initialization. Custom models may supply a zero-arg factory
        or new_instance(); standard module templates reset leaf parameters."""
        torch.manual_seed(self.seed if fit_seed is None else fit_seed)
        if not isinstance(template, nn.Module):
            model = template()
        elif hasattr(template, "new_instance"):
            model = template.new_instance()
        else:
            model = deepcopy(template)
            for module in model.modules():
                if not list(module.children()) and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                elif list(module.parameters(recurse=False)):
                    raise ValueError("custom parameter initialization requires a model factory or new_instance()")
        return model.to(self.device)


    def make_sampler(self, train_pairs, heldout_pairs, emb_df, fold=0, repetition=0,
                     excluded_drugs=()):
        """Construct a schedule before feature extraction; fail rather than
        silently claim epoch resampling from a previously downsampled cohort."""
        if self.positive_pool is None:
            if self.positive_sampling == "per_epoch":
                raise ValueError("per_epoch requires positive_pool: pass the complete eligible "
                                 "positive table to FusionTrainer, or select positive_sampling='fixed'")
            if self.positive_to_negative_ratio != 1.0:
                raise ValueError("a non-default ratio requires positive_pool")
            return None  # explicit legacy fixed-cohort fit
        pool = self.positive_pool
        ids = set(emb_df.index.astype(str))
        pool = pool[pool.drug1.isin(ids) & pool.drug2.isin(ids)]
        return EpochPositiveSampler(
            pool, train_pairs[train_pairs.label == 0], heldout_pairs,
            seed=self.seed, fold=fold, repetition=repetition,
            mode=self.positive_sampling,
            positive_to_negative_ratio=self.positive_to_negative_ratio,
            excluded_drugs=excluded_drugs)


    @staticmethod
    def sampler_drugs(sampler, train_pairs):
        """Cover every eligible endpoint, so later epochs cannot get zero-filled features."""
        drugs = set(train_pairs.drug1) | set(train_pairs.drug2)
        if sampler is not None:
            drugs.update(sampler.positives.ravel())
        return drugs


    def sampled_loaders(self, sampler, emb_df, val_pairs, test_pairs=None, *, scaler_epochs=None):
        """Fit the MeSH scaler on epoch-zero TRAINING pairs, then freeze it.
        Both sampling modes share the same epoch-zero draw and scaling policy.

        ``scaler_epochs=n`` fits the scaler incrementally on the whole n-epoch
        training schedule instead (still training pairs only). A feature that
        is near-constant in one 3,818-pair draw but not in the population gets
        a scale of ~1e-6 from epoch zero alone, and pairs drawn in later epochs
        then enter the network standardized to 1e5 -- which corrupts the
        BatchNorm running statistics used at evaluation time (Intermediate
        MeSH scope on the common-coverage cohort, 2026-09-18)."""
        X0, y0, p0 = build_pair_features(emb_df, sampler.epoch(0))
        if len(p0) != sampler.epoch_size:
            raise ValueError("sampling schedule contains drugs without embeddings")
        scaler = StandardScaler()
        if scaler_epochs is None:
            scaler.fit(X0)
        else:
            scaler.partial_fit(X0)
            for epoch in range(1, int(scaler_epochs)):
                Xe, _, pe = build_pair_features(emb_df, sampler.epoch(epoch))
                if len(pe) != sampler.epoch_size:
                    raise ValueError("sampling schedule contains drugs without embeddings")
                scaler.partial_fit(Xe)

        def loader(frame):
            X, y, kept = build_pair_features(emb_df, frame)
            if len(kept) != len(frame):
                raise ValueError("evaluation/sampled pairs lack embeddings")
            return DataLoader(DDIDataset(scaler.transform(X), y),
                              batch_size=self.batch_size, shuffle=False)

        def epoch_loader(epoch):
            return loader(sampler.epoch(epoch))

        return (epoch_loader(0), loader(val_pairs),
                None if test_pairs is None else loader(test_pairs), epoch_loader, scaler)


    def _make_stopper(self, patience=None):
        return EarlyStopping(
            patience=self.patience if patience is None else patience,
            min_delta=self.min_delta,
        )


    def _should_stop(self, val_losses, patience=None):
        """Stateless view of the early-stopping rule: replay `val_losses`
        through an EarlyStopping counter and report whether it would stop
        after the last epoch. The training loops use the counter directly."""
        stopper = self._make_stopper(patience)
        for l in val_losses:
            stopper.step(l)
        return stopper.should_stop


    def _evaluate_bins_topo_only(self, model, topo_df, test_pairs, mu, sigma, topo_cols):
        from dtpkg.metrics import evaluate_pair_scores
        validate_feature_coverage(topo_df, pair_drugs(test_pairs), "topology")
        model.eval()
        scores = []
        def feature(drug):
            a = (topo_df.loc[str(drug), topo_cols].to_numpy(float) - mu) / sigma
            return torch.tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            for row in test_pairs.itertuples(index=False):
                scores.append(torch.sigmoid(model(feature(row.drug1), feature(row.drug2))).item())
        bins, self.last_predictions = evaluate_pair_scores(test_pairs, scores, self.drug_to_cat)
        # Backward-compatible keys for older topology-only plotting callers.
        for metrics in bins.values():
            metrics.update(precision=metrics["prec"], recall=metrics["rec"])
        self.last_gates = None
        return bins


    def _initial_gate_records(self, model, pairs, topo_df, emb_df):
        if pairs is None or not hasattr(model, "fuse_one"):
            return pd.DataFrame()
        validate_feature_coverage(emb_df, pair_drugs(pairs), "MeSH")
        validate_feature_coverage(topo_df, pair_drugs(pairs), "topology")
        model.eval()
        rows = []
        with torch.no_grad():
            for drug in sorted(set(pairs.drug1) | set(pairs.drug2)):
                bio = torch.tensor(emb_df.loc[drug].to_numpy(), dtype=torch.float32, device=self.device)[None]
                topo = torch.tensor(topo_df.loc[drug].to_numpy(), dtype=torch.float32, device=self.device)[None]
                _, gate = model.fuse_one(bio, topo)
                if gate is None:
                    continue
                vals = gate.cpu().numpy().ravel()
                rows.append(dict(drug_id=drug, gate_mean=float(vals.mean()),
                    gate_sd_dimensions=float(vals.std()), gate_q25_dimensions=float(np.quantile(vals,.25)),
                    gate_q75_dimensions=float(np.quantile(vals,.75)), latent_dimensions=len(vals), phase="initial"))
        return pd.DataFrame(rows)


    def _measure_gates_on_pairs(self, model, pairs_df, topo_df, emb_df):
        validate_feature_coverage(emb_df, pair_drugs(pairs_df), "MeSH")
        validate_feature_coverage(topo_df, pair_drugs(pairs_df), "topology")
        model.eval()

        topo_dict = topo_df.to_dict(orient="index")
        emb_dict = emb_df.to_dict(orient="index")

        bio_dim = emb_df.shape[1]
        topo_dim = topo_df.shape[1]
        emb_cols = list(emb_df.columns)
        topo_cols = list(topo_df.columns)

        def get_feat(drug, src, dim, cols):
            if drug not in src:
                raise ValueError(f"Missing features for drug {drug!r}; zero imputation is disabled")
            return torch.tensor([src[drug][c] for c in cols], dtype=torch.float32)

        gate_vals = []
        per_drug = {}

        with torch.no_grad():
            for _, row in pairs_df.iterrows():
                d1, d2 = row["drug1"], row["drug2"]

                bio1 = get_feat(d1, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                bio2 = get_feat(d2, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                topo1 = get_feat(d1, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                topo2 = get_feat(d2, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)

                _, (g1, g2) = model(bio1, topo1, bio2, topo2, 0.0, 0.0)  # no dropout for eval
                if g1 is None or g2 is None:
                    continue
                topo_w = float((g1.mean() + g2.mean()) / 2)
                gate_vals.append(topo_w)
                for drug, g in ((d1, g1), (d2, g2)):
                    values = g.detach().cpu().numpy().ravel()
                    per_drug[str(drug)] = {"drug_id": str(drug),
                        "gate_mean": float(values.mean()), "gate_sd_dimensions": float(values.std()),
                        "gate_q25_dimensions": float(np.quantile(values, .25)),
                        "gate_q75_dimensions": float(np.quantile(values, .75)),
                        "latent_dimensions": len(values)}

        if not gate_vals:
            self.last_gates = None
            return None, None
        arr = np.array(gate_vals)
        self.last_gates = pd.DataFrame(per_drug.values())
        return float(arr.mean()), float(arr.std())


    def _train_fusion(
        self,
        model,
        train_pairs, val_pairs,
        topo_df, emb_df, drug_to_cat,
        *,
        test_pairs=None,
        patience=None, max_epochs=None,
        return_gate=False,
        epoch_sampler=None, fit_seed=None,
    ):
        """Fit on `train_pairs`, early-stop / select the checkpoint on
        `val_pairs`, then score `test_pairs` once with the restored best model.

        `test_pairs` is keyword-only so that a legacy positional call
        `(model, train, test, ...)` cannot silently turn the test fold into the
        stopping set. With `test_pairs=None` nothing is scored and the bin
        results (and gate summary) are None; the fitted model is still
        returned, which is what the inductive protocol needs.
        """
        # None -> the limits declared on the trainer (one source of truth)
        max_epochs = self.epochs if max_epochs is None else max_epochs
        required_drugs = self.sampler_drugs(epoch_sampler, train_pairs) | pair_drugs(val_pairs)
        if test_pairs is not None:
            required_drugs |= pair_drugs(test_pairs)
        validate_feature_coverage(topo_df, required_drugs, "topology")
        validate_feature_coverage(emb_df, required_drugs, "MeSH")
        stopper = self._make_stopper(patience)
        model = self._fresh_model(model, fit_seed)
        self.last_predictions = self.last_gates = None
        self.last_epoch_losses = []
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        self.initial_model_state = deepcopy(model.state_dict())
        initial_gates = self._initial_gate_records(model, test_pairs, topo_df, emb_df)

        # ---- lookup dicts ----
        topo_dict = topo_df.to_dict(orient="index")
        emb_dict = emb_df.to_dict(orient="index")

        bio_dim = emb_df.shape[1]
        topo_dim = topo_df.shape[1]
        emb_cols = list(emb_df.columns)
        topo_cols = list(topo_df.columns)

        def get_feat(drug, src, dim, cols):
            if drug not in src:
                raise ValueError(f"Missing features for drug {drug!r}; zero imputation is disabled")
            return torch.tensor([src[drug][c] for c in cols], dtype=torch.float32)

        # tracking
        val_losses = []
        best_state = None

        # ============================
        # TRAIN LOOP
        # ============================
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []
            train_by_class = {0: [], 1: []}

            # ---- train ----
            epoch_pairs = train_pairs if epoch_sampler is None else epoch_sampler.epoch(epoch)
            for _, row in epoch_pairs.iterrows():
                d1, d2, y = row["drug1"], row["drug2"], row["label"]

                bio1 = get_feat(d1, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                bio2 = get_feat(d2, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                topo1 = get_feat(d1, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                topo2 = get_feat(d2, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                yb = torch.tensor([y], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                logits, _ = model(bio1, topo1, bio2, topo2, self.drop_bio_prob, self.drop_topo_prob)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                train_by_class[int(y)].append(loss.item())

            # ============================
            # VALIDATION (inner split; the outer test fold is never seen here)
            # ============================
            model.eval()
            val_losses_epoch = []
            val_by_class = {0: [], 1: []}

            with torch.no_grad():
                for _, row in val_pairs.iterrows():
                    d1, d2, y = row["drug1"], row["drug2"], row["label"]

                    bio1 = get_feat(d1, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                    bio2 = get_feat(d2, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                    topo1 = get_feat(d1, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                    topo2 = get_feat(d2, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                    yb = torch.tensor([y], dtype=torch.float32).to(self.device)

                    logits, (_, _) = model(bio1, topo1, bio2, topo2, 0.0, 0.0)  # no dropout for eval
                    loss_value = criterion(logits, yb).item()
                    val_losses_epoch.append(loss_value)
                    val_by_class[int(y)].append(loss_value)

            self.last_epoch_losses.append(self._class_loss_row(epoch, train_by_class, val_by_class))
            mean_val_loss = np.mean(val_losses_epoch)
            val_losses.append(mean_val_loss)

            # Restore this checkpoint before measuring held-out gates.
            if stopper.step(mean_val_loss):
                best_state = deepcopy(model.state_dict())

            if stopper.should_stop:
                print(f"Early stop at epoch {epoch+1} "
                      f"(best val loss {stopper.best:.4f} at epoch {stopper.best_epoch+1})")
                break

        # RESTORE BEST MODEL
        if best_state is not None:
            model.load_state_dict(best_state)

        self.last_fit_info = {
            "epochs_run": len(val_losses),
            "best_epoch": stopper.best_epoch + 1,
            "best_val_loss": float(stopper.best),
            "n_train": len(train_pairs), "n_val": len(val_pairs),
            "n_test": None if test_pairs is None else len(test_pairs),
            "positive_sampling": "fixed" if epoch_sampler is None else epoch_sampler.mode,
            "fit_seed": self.seed if fit_seed is None else fit_seed,
        }
        if epoch_sampler is not None:
            self.last_fit_info.update(epoch_sampler.describe(len(val_losses)))
            self.last_fit_info["n_train"] = epoch_sampler.epoch_size
            self.last_fit_info["unique_training_positives"] = epoch_sampler.unique_positives(len(val_losses))

        if test_pairs is None:
            return (None, None, model) if return_gate else (None, model)

        # Outer test fold: scored once, with the validation-selected model.
        mean_topo, std_topo = self._measure_gates_on_pairs(
            model=model,
            pairs_df=test_pairs,
            topo_df=topo_df,
            emb_df=emb_df
        )

        gate_vals_summary = None
        if self.last_gates is not None:
            self.last_gates["phase"] = "trained"
            self.last_gates = pd.concat([initial_gates, self.last_gates], ignore_index=True)
            gate_vals_summary = {"mean_topo": mean_topo, "std_topo": std_topo}

        bin_results, self.last_predictions = evaluate_by_pair_bins_fusion(
            model=model,
            pairs_df=test_pairs,
            emb_df=emb_df,
            topo_df=topo_df,
            drug_to_cat=drug_to_cat,
            device=self.device,
            return_predictions=True,
        )

        if return_gate:
            return bin_results, gate_vals_summary, model
        else:
            return bin_results, model


    def _train_baseline(
        self, train_loader, val_loader,
        *,
        test_loader=None, test_pairs=None, drug_to_cat=None,
        patience=None, max_epochs=None, epoch_loader=None, fit_seed=None,
        epoch_sampler=None,
    ):
        """Fit on `train_loader`, early-stop / select on `val_loader`, then
        score `test_loader`/`test_pairs` once with the restored best model.
        The test arguments are keyword-only (see `_train_fusion`); when they
        are omitted nothing is scored and the bin results are None."""
        if (test_loader is None) != (test_pairs is None):
            raise ValueError("pass both test_loader and test_pairs, or neither")
        if test_pairs is not None and drug_to_cat is None:
            raise ValueError("drug_to_cat is required to score test_pairs by pair bin")
        # None -> the limits declared on the trainer (one source of truth)
        max_epochs = self.epochs if max_epochs is None else max_epochs
        stopper = self._make_stopper(patience)
        model = self._fresh_model(self.baseline_cls, fit_seed)
        self.last_predictions = self.last_gates = None
        self.last_epoch_losses = []
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        val_losses, best_state = [], None

        for epoch in range(max_epochs):
            # ---- Train ----
            model.train()
            train_by_class = {0: [], 1: []}
            current_loader = train_loader if epoch_loader is None else epoch_loader(epoch)
            for xb, yb in current_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                per_example = nn.functional.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb, reduction="none")
                loss = per_example.mean()
                for label in (0, 1):
                    train_by_class[label].extend(per_example[yb == label].detach().cpu().tolist())
                loss.backward()
                optimizer.step()

            # ---- Validate (inner split; the outer test fold is never seen here) ----
            model.eval()
            losses = []
            val_by_class = {0: [], 1: []}
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    per_example = nn.functional.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb, reduction="none")
                    losses.extend(per_example.cpu().tolist())
                    for label in (0, 1):
                        val_by_class[label].extend(per_example[yb == label].cpu().tolist())
            self.last_epoch_losses.append(self._class_loss_row(epoch, train_by_class, val_by_class))
            mean_val_loss = np.mean(losses)
            val_losses.append(mean_val_loss)

            if stopper.step(mean_val_loss):
                best_state = deepcopy(model.state_dict())

            if stopper.should_stop:
                print(f"Early stop at epoch {epoch+1} "
                      f"(best val loss {stopper.best:.4f} at epoch {stopper.best_epoch+1})")
                break

        model.load_state_dict(best_state)

        self.last_fit_info = {
            "epochs_run": len(val_losses),
            "best_epoch": stopper.best_epoch + 1,
            "best_val_loss": float(stopper.best),
            "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
            "n_test": None if test_pairs is None else len(test_pairs),
            "positive_sampling": "fixed" if epoch_sampler is None else epoch_sampler.mode,
            "fit_seed": self.seed if fit_seed is None else fit_seed,
        }
        if epoch_sampler is not None:
            self.last_fit_info.update(epoch_sampler.describe(len(val_losses)))
            self.last_fit_info["unique_training_positives"] = epoch_sampler.unique_positives(len(val_losses))

        if test_loader is None:
            return None, model

        # ---- Outer test fold: scored once, with the validation-selected model ----
        # Pass the trainer's device: the evaluator otherwise picks CUDA whenever it
        # is visible, even for a CPU-resident model.
        bin_results, self.last_predictions = evaluate_by_pair_bins(
            model, test_loader, test_pairs, drug_to_cat, device=self.device, return_predictions=True)
        print("Baseline evaluation:", {k: v["auc"] for k, v in bin_results.items()})
        return bin_results, model


    def _topo_for_mask(self, extractor, masked_pairs, all_drugs, cache):
        """Extract once per distinct (masked positive pairs, drug set) and
        reuse the table for every (repetition, fold) that shares that mask.

        Keyed by the actual mask rather than the fold index, so hoisting stays
        correct whether the inner validation assignment is shared across
        repetitions (one extraction per fold) or varies (one per distinct
        mask). Both validation and test positive targets are removed before
        extraction."""
        key = masked_pairs_key(masked_pairs, all_drugs)
        if key not in cache:
            if hasattr(extractor, "G"):
                missing = sorted(set(all_drugs) - set(extractor.G))
                if missing:
                    raise ValueError(f"Topology graph is missing {len(missing)} requested drugs: {missing[:10]}")
            topo_df = extractor.compute_for_fold(all_drugs, eval_pairs=masked_pairs)
            validate_feature_coverage(topo_df, all_drugs, "topology")
            diagnostics = getattr(extractor, "last_diagnostics", None)
            if isinstance(diagnostics, pd.DataFrame) and "status" in diagnostics:
                failures = diagnostics[diagnostics.status.isin(["missing_node", "error"])]
                if not failures.empty:
                    raise ValueError(f"Topology extraction failed: {failures.head(10).to_dict('records')}")
            # A genuine isolate has a computed feature row (including ego size);
            # missing nodes and extraction failures must never become zero rows.
            cache[key] = (topo_df.loc[list(all_drugs)].copy(), diagnostics)
        validate_feature_coverage(cache[key][0], all_drugs, "topology")
        return cache[key]


    def run_experiments(self, X, y, pairs, folds, emb_df, drug_to_cat,
                        num_experiments=3, splits=None, checkpoint_dir=None):
        """
        Run experiments for:
        - Baseline (MeSH only)
        - Fusion (MeSH + Topo)
        - Fusion_with_GO (MeSH + Topo + GO)  optional, only if topo_extractor_with_go is provided

        Protocol per (repetition, fold): the outer training fold is split into
        an inner training set and a stratified validation set (`self.val_frac`)
        *before* anything else; validation and test positive targets are both
        masked from the graph before topology extraction; the MeSH scaler is
        fitted on the inner training rows; models early-stop / select their
        checkpoint on validation; the outer test fold is scored exactly once.
        `splits` (``splits[rep][fold] == (train_idx, val_idx, test_idx)``, e.g.
        from `MeSHDataLoaders.load_split_manifest`) lets paired graph arms
        reuse one assignment; by default it is derived from `self.seed`. The
        assignment is written to `<out_dir>/split_manifest.json`.
        """
        self.results = {}
        self.training_log = []
        prediction_frames, gate_frames, epoch_loss_rows = [], [], []
        self.drug_to_cat = drug_to_cat
        validate_drug_depths(drug_to_cat, pair_drugs(pairs))

        if splits is None:
            splits = create_inner_splits(
                y, folds, val_frac=self.val_frac, seed=self.seed,
                num_repetitions=num_experiments,
                vary_by_repetition=self.inner_val_per_repetition,
                repeat_outer=self.repeat_outer_splits)
        elif len(splits) < num_experiments:
            raise ValueError(f"splits has {len(splits)} repetitions, "
                             f"num_experiments={num_experiments}")
        self.splits = splits
        # Group identical outer partitions; training seeds do not create new CV repeats.
        partitions = {}
        self.split_repeat_ids = []
        for split in splits:
            key = tuple(tuple(sorted(map(int, te))) for _, _, te in split)
            self.split_repeat_ids.append(partitions.setdefault(key, len(partitions)))
        save_split_manifest(splits, pairs,
                            os.path.join(self.out_dir, "split_manifest.json"),
                            seed=self.seed, val_frac=self.val_frac,
                            extra={"positive_sampling": self.positive_sampling,
                                   "positive_to_negative_ratio": self.positive_to_negative_ratio,
                                   "train_indices_describe": "evaluation-cohort training partition; "
                                   "sampled training positives may also come from positive_pool"})

        # === Topological features depend on the actual mask, not on the repetition ===
        # Extraction is hoisted out of the (repetition, fold) loop and keyed by
        # the masked positive pairs + drug set (`_topo_for_mask`); with a shared
        # inner split that is one extraction per fold, as before. Repetitions
        # still get fresh model initialisation, shuffling and training.
        topo_cache, topo_cache_go = {}, {}
        samplers = {}

        def fold_pairs(rep, fold_idx):
            tr_idx, va_idx, te_idx = splits[rep][fold_idx]
            train_pairs = pairs.iloc[tr_idx].reset_index(drop=True)
            val_pairs = pairs.iloc[va_idx].reset_index(drop=True)
            test_pairs = pairs.iloc[te_idx].reset_index(drop=True)
            masked_pairs = pd.concat([val_pairs, test_pairs], ignore_index=True)
            if (rep, fold_idx) not in samplers:
                samplers[(rep, fold_idx)] = self.make_sampler(
                    train_pairs, masked_pairs, emb_df, fold_idx, rep)
            all_drugs = sorted(
                self.sampler_drugs(samplers[(rep, fold_idx)], train_pairs)
                | set(masked_pairs["drug1"]) | set(masked_pairs["drug2"])
            )
            return (tr_idx, va_idx, te_idx), train_pairs, val_pairs, test_pairs, masked_pairs, all_drugs

        print(f"\n=== Extracting topological features for {len(folds)} folds × "
              f"{num_experiments} repetitions (once per distinct val+test mask) ===")
        for exp in range(num_experiments):
            for fold_idx in range(len(folds)):
                _, _, _, _, masked_pairs, all_drugs = fold_pairs(exp, fold_idx)
                if self.include_graph_baselines:
                    from dtpkg.fusion.graph_baselines import masked_ddi_graph
                    masked_ddi_graph(self.topo_extractor.G, masked_pairs)
                if masked_pairs_key(masked_pairs, all_drugs) not in topo_cache:
                    print(f"\n--- Repetition {exp+1} fold {fold_idx+1}/{len(folds)} topology "
                          f"(masking {int(masked_pairs['label'].sum())} val+test positives) ---")
                _, diag = self._topo_for_mask(self.topo_extractor, masked_pairs, all_drugs, topo_cache)
                self.topo_diagnostics[(exp, fold_idx)] = diag
                if getattr(self, "topo_extractor_with_go", None) is not None:
                    self._topo_for_mask(self.topo_extractor_with_go, masked_pairs, all_drugs, topo_cache_go)
        print(f"Extracted {len(topo_cache)} distinct masked tables.")

        for exp in range(num_experiments):
            print(f"\n=== Experiment {exp+1}/{num_experiments} ===")
            self.results[exp] = {}

            for fold_idx in range(len(folds)):
                print(f"\n--- Fold {fold_idx+1}/{len(folds)} ---")

                # === Split data (scaler fitted on the inner training rows only) ===
                (tr_idx, va_idx, te_idx), train_pairs, val_pairs, test_pairs, masked_pairs, all_drugs = \
                    fold_pairs(exp, fold_idx)
                tr_loader, va_loader, te_loader = get_split_loaders(
                    X, y, tr_idx, va_idx, te_idx, batch_size=self.batch_size
                )
                sampler = samplers[(exp, fold_idx)]
                epoch_loader = None
                # The MeSH scaler is part of the fitted baseline; keep it for the bundle.
                scaler = StandardScaler().fit(X[tr_idx])
                if sampler is not None:
                    tr_loader, va_loader, te_loader, epoch_loader, scaler = self.sampled_loaders(
                        sampler, emb_df, val_pairs, test_pairs)
                fit_seed = epoch_seed(self.seed, fold_idx, exp, 0)
                topo_df, _ = self._topo_for_mask(self.topo_extractor, masked_pairs, all_drugs, topo_cache)
                topo_df_with_go = None
                if getattr(self, "topo_extractor_with_go", None) is not None:
                    topo_df_with_go, _ = self._topo_for_mask(
                        self.topo_extractor_with_go, masked_pairs, all_drugs, topo_cache_go)

                def log(model_name):
                    metadata = self._fold_metadata(exp, fold_idx)
                    self.training_log.append({"experiment": exp, "fold": fold_idx,
                                              "model": model_name, **self.last_fit_info, **metadata})
                    if self.last_predictions is not None:
                        p = self.last_predictions.rename(columns={"pred": "score", "pair_bin": "category"}).copy()
                        if "threshold" not in p: p["threshold"] = .5
                        if "score_kind" not in p: p["score_kind"] = "neural_probability"
                        prediction_frames.append(p.assign(model=model_name, experiment=exp, **metadata))
                    epoch_loss_rows.extend(dict(model=model_name, **metadata, **r) for r in self.last_epoch_losses)
                    if self.last_gates is not None:
                        g = self.last_gates.copy()
                        g["depth"] = g.drug_id.map(drug_to_cat)
                        gate_frames.append(g.assign(model=model_name, experiment=exp, **metadata))

                # === Train models ===
                # Fitted models, their fit records and every transform used to score
                # the test fold are bundled per (repetition, fold) when
                # `checkpoint_dir` is given, so later tasks reuse these fits instead
                # of retraining (same bundle layout as run_inductive_eval).
                checkpoint_models, checkpoint_fit_info, checkpoint_epoch_losses = {}, {}, {}

                def keep(model_name, model):
                    checkpoint_models[model_name] = model
                    checkpoint_fit_info[model_name] = deepcopy(self.last_fit_info)
                    checkpoint_epoch_losses[model_name] = deepcopy(self.last_epoch_losses)

                print("\n→ Training baseline model (MeSH only)")
                baseline_bins, baseline_model = self._train_baseline(
                    tr_loader, va_loader,
                    test_loader=te_loader, test_pairs=test_pairs, drug_to_cat=drug_to_cat,
                    epoch_loader=epoch_loader, epoch_sampler=sampler, fit_seed=fit_seed)
                log("baseline")
                keep("baseline", baseline_model)

                print("\n→ Training fusion model (MeSH + Topo)")
                fusion_bins_wo_go, fusion_model = self._train_fusion(
                    self.fusion_model_wo_go, train_pairs, val_pairs, topo_df, emb_df, drug_to_cat,
                    test_pairs=test_pairs, epoch_sampler=sampler, fit_seed=fit_seed)
                log("fusion")
                keep("fusion", fusion_model)

                # Optional 3rd model
                if topo_df_with_go is not None:
                    print("\n→ Training fusion_with_GO model (MeSH + Topo + GO)")
                    fusion_go_bins, _ = self._train_fusion(
                        self.fusion_model_go, train_pairs, val_pairs, topo_df_with_go, emb_df, drug_to_cat,
                        test_pairs=test_pairs, epoch_sampler=sampler, fit_seed=fit_seed)
                    log("fusion_with_go")
                else:
                    fusion_go_bins = None

                model_bins = {"baseline": baseline_bins, "fusion": fusion_bins_wo_go}
                if fusion_go_bins is not None:
                    model_bins["fusion_with_go"] = fusion_go_bins
                if self.include_topology_baseline:
                    from dtpkg.fusion.topology_baseline import TopologyBaseline, TopoOnlyModel
                    ablation = TopologyBaseline(self)
                    model_bins["topo_only"] = ablation._train_topo_only(
                        TopoOnlyModel(topo_df.shape[1]), train_pairs, val_pairs, topo_df,
                        test_pairs=test_pairs, epoch_sampler=sampler, fit_seed=fit_seed,
                        device=self.device, lr=self.lr, weight_decay=self.weight_decay)
                    log("topo_only")
                    keep("topo_only", ablation.last_model)
                    topo_only_normalization = ablation.last_normalization
                else:
                    topo_only_normalization = None
                graph_thresholds = {}
                if self.include_graph_baselines:
                    from dtpkg.fusion.graph_baselines import masked_ddi_graph, graph_scores, validation_f1_threshold
                    from dtpkg.metrics import evaluate_pair_scores
                    graph = masked_ddi_graph(self.topo_extractor.G, masked_pairs)
                    for name in ("common_neighbors", "degree_product"):
                        threshold = validation_f1_threshold(val_pairs.label, graph_scores(graph, val_pairs, name))
                        model_bins[name], self.last_predictions = evaluate_pair_scores(
                            test_pairs, graph_scores(graph, test_pairs, name), drug_to_cat, threshold)
                        self.last_predictions["score_kind"] = "raw_graph_score"
                        self.last_gates, self.last_epoch_losses = None, []
                        self.last_fit_info = dict(epochs_run=0, n_train=len(train_pairs), n_val=len(val_pairs),
                            n_test=len(test_pairs), threshold=threshold, threshold_policy="inner_validation_F1",
                            positive_sampling="not_applicable")
                        log(name)
                        graph_thresholds[name] = threshold
                if checkpoint_dir is not None:
                    self._save_cv_fit_bundle(
                        checkpoint_dir, exp, fold_idx, checkpoint_models, checkpoint_fit_info,
                        checkpoint_epoch_losses, scaler=scaler, emb_df=emb_df, topo_df=topo_df,
                        topo_only_normalization=topo_only_normalization, graph_thresholds=graph_thresholds,
                        drug_to_cat=drug_to_cat, train_pairs=train_pairs, val_pairs=val_pairs,
                        test_pairs=test_pairs, sampler=sampler)
                all_bins = set().union(*(bins.keys() for bins in model_bins.values()))
                fold_results = {name: {model: bins.get(name, {}) for model, bins in model_bins.items()}
                                for name in all_bins}

                self.results[exp][fold_idx] = fold_results

        # === Save results ===
        self._save_detailed_results()
        self.oof_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        self.gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
        self.oof_predictions_df.to_csv(os.path.join(self.out_dir, "oof_predictions.csv"), index=False)
        self.gates_df.to_csv(os.path.join(self.out_dir, "heldout_drug_gates.csv"), index=False)
        pd.DataFrame(self.training_log).to_csv(
            os.path.join(self.out_dir, "training_log.csv"), index=False)
        pd.DataFrame(epoch_loss_rows).to_csv(os.path.join(self.out_dir, "epoch_class_losses.csv"), index=False)
        print("\nAll experiments completed and saved.")


    def _save_cv_fit_bundle(self, checkpoint_dir, exp, fold, models, fit_info, epoch_losses, *,
                            scaler, emb_df, topo_df, topo_only_normalization, graph_thresholds,
                            drug_to_cat, train_pairs, val_pairs, test_pairs, sampler):
        """Write `<checkpoint_dir>/fit_<rep>_<fold>.pt`: the restored best models of
        one (repetition, fold) plus everything needed to score its test fold again
        without training. Trusted local pickle (full modules); load with
        `load_cv_fit`. Written atomically so an interrupted job leaves no partial file."""
        from pathlib import Path
        destination = Path(checkpoint_dir) / f"fit_{exp:02d}_{fold:02d}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        bundle = dict(
            format_version=1, protocol="cv",
            models={k: deepcopy(v).cpu() for k, v in models.items()},
            baseline_scaler=scaler, embeddings=emb_df,
            # Raw (unscaled) fold table the fusion model trained and was scored on;
            # topology-only standardises it with `topo_only_normalization`.
            eval_topology=topo_df, topo_only_normalization=topo_only_normalization,
            graph_thresholds=graph_thresholds, drug_to_cat=drug_to_cat,
            train_pairs=train_pairs, validation_pairs=val_pairs, test_pairs=test_pairs,
            metadata={**self._fold_metadata(exp, fold), "experiment": exp},
            fit_info=fit_info, epoch_class_losses=epoch_losses,
            sampling_config={**dict(seed=self.seed, fold=fold, repetition=exp,
                                    mode=self.positive_sampling,
                                    positive_to_negative_ratio=self.positive_to_negative_ratio),
                             **({} if sampler is None else sampler.describe())},
            trainer_config=dict(epochs=self.epochs, patience=self.patience, min_delta=self.min_delta,
                                lr=self.lr, weight_decay=self.weight_decay, batch_size=self.batch_size,
                                drop_bio_prob=self.drop_bio_prob, drop_topo_prob=self.drop_topo_prob,
                                val_frac=self.val_frac),
        )
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(bundle, temporary)
        temporary.replace(destination)
        return destination


    def _fold_metadata(self, exp, fold):
        tr, va, te = self.splits[exp][fold]
        return {"split_repeat": self.split_repeat_ids[exp], "fold": fold,
                "training_seed": epoch_seed(self.seed, fold, exp, 0),
                "inference_protocol": "cv", "n_outer_train": len(tr) + len(va),
                "n_outer_test": len(te)}


    def _save_detailed_results(self, filename=None):
        """
        Save the detailed results of all experiments (nested dict self.results)
        into a structured DataFrame and optionally a pickle file.
        """
        import pandas as pd, os, pickle

        rows = []
        for exp_id, folds in self.results.items():
            for fold_id, bins in folds.items():
                for bin_name, models in bins.items():
                    for model_name, metrics in models.items():
                        row = {"experiment": exp_id, "fold": fold_id, "bin": bin_name, "model": model_name}
                        row.update(metrics)
                        row.update(self._fold_metadata(exp_id, fold_id))
                        rows.append(row)

        df = pd.DataFrame(rows)
        self.results_df = df  # attach for later analysis
        df.to_csv(os.path.join(self.out_dir, "fold_metrics.csv"), index=False)

        if filename is None:
            os.makedirs(self.out_dir, exist_ok=True)
            filename = os.path.join(self.out_dir, "detailed_results.pkl")

        with open(filename, "wb") as f:
            pickle.dump(self.results, f)

        print(f"Saved detailed results to {filename}")
        print(f"DataFrame shape: {df.shape}")
        return df

    def _split_train_val(self, X, y, pairs, val_split, repetition=0):
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset, DataLoader

        # --- split indexes ---
        idx_train, idx_val = train_test_split(
            np.arange(len(y)),
            test_size=val_split,
            random_state=self.seed + 1000 * repetition,
            stratify=y
        )

        # --- split arrays ---
        X_train, X_val = X[idx_train], X[idx_val]
        y_train, y_val = y[idx_train], y[idx_val]

        # --- split pairs ---
        train_pairs_df = pairs.iloc[idx_train].reset_index(drop=True)
        val_pairs_df   = pairs.iloc[idx_val].reset_index(drop=True)

        # --- build loaders ---
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, train_pairs_df, val_pairs_df


    def run_inductive_eval(
        self,
        X, y, pairs,
        emb_df,
        topo_inductive_df,
        inductive_test_pairs,
        drug_to_cat,
        num_experiments=5,
        val_split=0.15,
        inductive_drugs=None,
        heldout_topology_factory=None,
        checkpoint_dir=None,
        scaling_policy="epoch_zero",
        checkpoint_identity=None,
        fit_baseline=True,
        graph_baseline_factory=None,
    ):
        """`fit_baseline=False` skips the MeSH-only fit so a paired arm can reuse
        the reference arm's saved baseline predictions instead of refitting under
        merely matching seeds. `graph_baseline_factory(masked_pairs)` returns
        `{name: scorer(pairs)}` raw graph scores on the inference graph; each is
        thresholded on inner validation only and scored once on the test pairs."""

        print("\n========================================================")
        print("               Running Inductive Evaluation             ")

        if scaling_policy not in ("epoch_zero", "planned_training_schedule"):
            raise ValueError("unknown inductive scaling policy")

        self.inductive_results = []
        prediction_frames, gate_frames = [], []
        self.drug_to_cat = drug_to_cat
        validate_drug_depths(drug_to_cat, pair_drugs(pairs) | pair_drugs(inductive_test_pairs))
        if inductive_drugs is None:
            raise ValueError("pass the full inductive_drugs set to exclude its endpoints from sampling")
        excluded = set() if inductive_drugs is None else set(map(str, inductive_drugs))
        if excluded & (set(pairs.drug1) | set(pairs.drug2)):
            raise ValueError("training/validation cohort contains inductive drugs")
        if not (inductive_test_pairs.drug1.isin(excluded) | inductive_test_pairs.drug2.isin(excluded)).all():
            raise ValueError("each inductive test pair must contain a declared held-out drug")
        test_drugs = pair_drugs(inductive_test_pairs)
        heldout_test_drugs = test_drugs & excluded
        known_test_drugs = test_drugs - excluded
        validate_feature_coverage(emb_df, pair_drugs(pairs) | test_drugs, "MeSH")
        if heldout_topology_factory is None:
            validate_feature_coverage(topo_inductive_df, heldout_test_drugs, "held-out topology")
        # The extractor normally returns held-out rows only. If a caller supplies
        # extra known rows, compute their masked features below instead of using them.
        heldout_topo = (None if heldout_topology_factory is not None else
                       topo_inductive_df.loc[sorted(heldout_test_drugs)])

        for exp in range(num_experiments):
            print(f"\n========== Inductive Experiment {exp+1}/{num_experiments} ==========\n")

            # STEP 1 — Train/Validation split
            train_loader, val_loader, train_pairs_df, val_pairs_df = \
                self._split_train_val(X, y, pairs, val_split, repetition=exp)

            if heldout_topology_factory is not None:
                # Recompute on the inference graph with THIS inner validation
                # mask, in addition to all held-out incident DDI exclusions.
                heldout_topo = heldout_topology_factory(val_pairs_df).loc[sorted(heldout_test_drugs)]
                validate_feature_coverage(heldout_topo, heldout_test_drugs, "held-out topology")

            sampler = self.make_sampler(
                train_pairs_df, pd.concat([val_pairs_df, inductive_test_pairs], ignore_index=True),
                emb_df, repetition=exp, excluded_drugs=excluded)
            epoch_loader = scaler = None
            if sampler is not None:
                train_loader, val_loader, _, epoch_loader, scaler = self.sampled_loaders(
                    sampler, emb_df, val_pairs_df,
                    scaler_epochs=self.epochs if scaling_policy == "planned_training_schedule" else None)
            fit_seed = epoch_seed(self.seed, 0, exp, 0)

            # STEP 2 — Compute train/val topo features
            # Identify all drugs used in train or val
            trainval_drugs = list(
                self.sampler_drugs(sampler, train_pairs_df)
                .union(val_pairs_df["drug1"]).union(val_pairs_df["drug2"])
                .union(known_test_drugs)
            )

            # Leakage-safe removal of val DDIs
            topo_trainval_df = self.topo_extractor.compute_for_fold(
                drug_ids=trainval_drugs,
                eval_pairs=val_pairs_df  # remove edges of val fold
            )
            validate_feature_coverage(topo_trainval_df, trainval_drugs, "known-drug topology")
            topo_trainval_df = topo_trainval_df.loc[trainval_drugs]
            if set(heldout_topo.columns) != set(topo_trainval_df.columns):
                raise ValueError("Known and held-out topology must have the same feature columns")

            # --------------------------
            # Normalize topo features
            # --------------------------
            # Compute statistics using *train-only* drugs
            scaling_pairs = train_pairs_df if sampler is None else sampler.epoch(0)
            train_drugs = sorted(sampler.endpoint_union(self.epochs) if
                sampler is not None and scaling_policy == "planned_training_schedule" else
                set(scaling_pairs.drug1) | set(scaling_pairs.drug2))

            topo_train_df = topo_trainval_df.loc[train_drugs]

            # mean / std per feature using training part only
            self.topo_mean = topo_train_df.mean(axis=0)
            self.topo_std = topo_train_df.std(axis=0) + 1e-8

            # apply normalization to train+val topo
            topo_trainval_df = (topo_trainval_df - self.topo_mean) / self.topo_std

            # normalize the inductive topo features using SAME stats
            topo_inductive_df_norm = (heldout_topo[topo_trainval_df.columns] - self.topo_mean) / self.topo_std
            eval_topo = pd.concat([topo_trainval_df, topo_inductive_df_norm], verify_integrity=True)
            validate_feature_coverage(eval_topo, test_drugs, "inductive evaluation topology")

            # --------------------------------------------------
            # STEP 3 — Train baseline
            # --------------------------------------------------
            # Validation is used for stopping/selection only; the inductive
            # test pairs below are scored once with the selected model.
            baseline_model = None
            if fit_baseline:
                print("Training Baseline (MeSH only)")
                _, baseline_model = self._train_baseline(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epoch_loader=epoch_loader, epoch_sampler=sampler, fit_seed=fit_seed,
                )
                baseline_fit_info = deepcopy(self.last_fit_info)
                baseline_epoch_losses = deepcopy(self.last_epoch_losses)
            else:
                print("MeSH-only baseline not fitted here; reference-arm predictions are reused")

            # --------------------------------------------------
            # STEP 4 — Train fusion model
            # --------------------------------------------------
            print("\nTraining Fusion Model (MeSH + Topo)")
            _, fusion_model = self._train_fusion(
                model=self.fusion_model_wo_go,
                train_pairs=train_pairs_df,
                val_pairs=val_pairs_df,
                topo_df=topo_trainval_df,
                emb_df=emb_df,
                drug_to_cat=drug_to_cat,
                epoch_sampler=sampler, fit_seed=fit_seed,
            )
            fusion_fit_info = deepcopy(self.last_fit_info)
            fusion_epoch_losses = deepcopy(self.last_epoch_losses)
            # --------------------------------------------------
            # 5. Inductive evaluation
            # --------------------------------------------------
            print("\nEvaluating on inductive test set...")

            model_bins, model_predictions = {}, {}
            checkpoint_models, checkpoint_fit_info, checkpoint_epoch_losses = {}, {}, {}
            if fit_baseline:
                model_bins["baseline"], model_predictions["baseline"] = evaluate_baseline_inductive(
                    model=baseline_model.to(self.device),
                    pairs_df=inductive_test_pairs,
                    emb_df=emb_df,
                    drug_to_cat=drug_to_cat,
                    device=self.device, scaler=scaler, return_predictions=True,
                )
                checkpoint_models["baseline"] = baseline_model
                checkpoint_fit_info["baseline"] = baseline_fit_info
                checkpoint_epoch_losses["baseline"] = baseline_epoch_losses

            # fusion
            model_bins["fusion"], model_predictions["fusion"] = evaluate_by_pair_bins_fusion(
                model=fusion_model.to(self.device),
                pairs_df=inductive_test_pairs,
                emb_df=emb_df,
                topo_df=eval_topo,
                drug_to_cat=drug_to_cat,
                device=self.device, return_predictions=True,
            )

            metadata = dict(split_repeat=0, fold=0, training_seed=fit_seed,
                inference_protocol="inductive_fixed_holdout", n_outer_train=len(pairs),
                n_outer_test=len(inductive_test_pairs), experiment=exp)
            # Actual initial and restored-checkpoint gates under the SAME test inputs.
            initial = deepcopy(fusion_model)
            initial.load_state_dict(self.initial_model_state)
            initial_gates = self._initial_gate_records(initial, inductive_test_pairs, eval_topo, emb_df)
            self._measure_gates_on_pairs(fusion_model, inductive_test_pairs, eval_topo, emb_df)
            gates = pd.DataFrame()
            if self.last_gates is not None:
                trained_gates = self.last_gates.assign(phase="trained")
                gates = pd.concat([initial_gates, trained_gates], ignore_index=True)
                # Retain held-out drugs only; training partners are not inductive gate observations.
                gates = gates[gates.drug_id.isin(excluded)].copy()
                gates["depth"] = gates.drug_id.map(drug_to_cat)
                gate_frames.append(gates.assign(model="fusion", **metadata))
            checkpoint_models["fusion"] = fusion_model
            checkpoint_fit_info["fusion"] = fusion_fit_info
            checkpoint_epoch_losses["fusion"] = fusion_epoch_losses
            topo_only_normalization = None
            if self.include_topology_baseline:
                from dtpkg.fusion.topology_baseline import TopologyBaseline, TopoOnlyModel
                ablation = TopologyBaseline(self)
                ablation._train_topo_only(TopoOnlyModel(topo_trainval_df.shape[1]), train_pairs_df,
                    val_pairs_df, topo_trainval_df, epoch_sampler=sampler, fit_seed=fit_seed,
                    device=self.device, lr=self.lr, weight_decay=self.weight_decay,
                    normalization_drugs=train_drugs if scaling_policy == "planned_training_schedule" else None)
                mu, sigma, cols = ablation.last_normalization
                model_bins["topo_only"] = self._evaluate_bins_topo_only(
                    ablation.last_model, eval_topo, inductive_test_pairs, mu, sigma, cols)
                model_predictions["topo_only"] = self.last_predictions.copy()
                checkpoint_models["topo_only"] = ablation.last_model
                topo_only_normalization = ablation.last_normalization
                checkpoint_fit_info["topo_only"] = deepcopy(self.last_fit_info)
                checkpoint_epoch_losses["topo_only"] = deepcopy(self.last_epoch_losses)
            graph_thresholds, graph_baseline_predictions = {}, {}
            if graph_baseline_factory is not None:
                from dtpkg.fusion.graph_baselines import validation_f1_threshold
                from dtpkg.metrics import evaluate_pair_scores
                # The threshold is chosen on inner validation only, scored on the
                # TRAINING graph (held-out nodes absent, validation positives masked)
                # so that held-out structure cannot touch calibration; test pairs are
                # scored on the inference graph. The ranking is never reversed.
                # Validation pairs join two development drugs, so the threshold is
                # not calibrated for held-out endpoints: rank metrics are the
                # primary reading of these raw scores.
                scorers = graph_baseline_factory(pd.concat([val_pairs_df, inductive_test_pairs], ignore_index=True))
                for name, scorer in scorers.items():
                    threshold = validation_f1_threshold(val_pairs_df.label, scorer["validation"](val_pairs_df))
                    model_bins[name], predictions = evaluate_pair_scores(
                        inductive_test_pairs, scorer["test"](inductive_test_pairs), drug_to_cat, threshold)
                    predictions["score_kind"] = "raw_graph_score"
                    model_predictions[name] = predictions
                    graph_baseline_predictions[name] = predictions.copy()
                    graph_thresholds[name] = threshold
                    checkpoint_fit_info[name] = dict(epochs_run=0, n_train=len(train_pairs_df),
                        n_val=len(val_pairs_df), n_test=len(inductive_test_pairs), threshold=threshold,
                        threshold_policy="inner_validation_F1", positive_sampling="not_applicable")
            if checkpoint_dir is not None:
                from pathlib import Path
                destination = Path(checkpoint_dir) / f"fit_{exp:02d}.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Trusted local inference bundle; full modules preserve exact
                # architectures, and every transform used in scoring is retained.
                bundle = dict(format_version=1, protocol="inductive_fixed_holdout",
                    fit_identity=checkpoint_identity, scaling_policy=scaling_policy,
                    topology_scaling_drugs=train_drugs,
                    models={k: deepcopy(v).cpu() for k, v in checkpoint_models.items()},
                    baseline_scaler=scaler, embeddings=emb_df, eval_topology=eval_topo,
                    topology_mean=self.topo_mean, topology_std=self.topo_std,
                    topo_only_normalization=topo_only_normalization,
                    heldout_drugs=sorted(excluded), drug_to_cat=drug_to_cat,
                    train_pairs=train_pairs_df, validation_pairs=val_pairs_df,
                    test_pairs=inductive_test_pairs, metadata=metadata,
                    fit_info=checkpoint_fit_info, epoch_class_losses=checkpoint_epoch_losses,
                    graph_thresholds=graph_thresholds, graph_baseline_predictions=graph_baseline_predictions,
                    gates=gates.assign(model="fusion", **metadata),
                    fold_metrics=pd.DataFrame([dict(model=name, category=bin_name, **metadata, **metrics)
                                               for name, bins in model_bins.items()
                                               for bin_name, metrics in bins.items()]),
                    sampling_config=dict(seed=self.seed, repetition=exp,
                        mode=self.positive_sampling, positive_to_negative_ratio=self.positive_to_negative_ratio))
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(bundle, temporary)
                temporary.replace(destination)
            if self.include_graph_baselines and graph_baseline_factory is None:
                # Legacy inductive path without an inference-graph scorer: every pair
                # includes a drug with no observed DDI edges, so CN and degree
                # product are structurally unavailable, not AUC=0.5 results.
                counts = model_bins["fusion"]
                for name in ("common_neighbors", "degree_product"):
                    model_bins[name] = {
                        category: {**{k: values[k] for k in ("n", "n_pos", "n_neg")},
                            **dict.fromkeys(("auc", "ap", "f1", "prec", "rec", "acc", "specificity",
                                             "tp", "fp", "tn", "fn"), np.nan),
                            "status": "not_applicable_no_ddi_edges"}
                        for category, values in counts.items()}
            for name, predictions in model_predictions.items():
                predictions = predictions.rename(columns={"pred": "score", "pair_bin": "category"})
                if "threshold" not in predictions: predictions["threshold"] = .5
                if "score_kind" not in predictions:
                    predictions["score_kind"] = "neural_probability"
                predictions["drug1_heldout"] = predictions.drug1.isin(excluded)
                predictions["drug2_heldout"] = predictions.drug2.isin(excluded)
                prediction_frames.append(predictions.assign(model=name, **metadata))
            for name, bins in model_bins.items():
                for bin_name, metrics in bins.items():
                    self.inductive_results.append(dict(model=name, category=bin_name, **metadata, **metrics))

        # produce final DF
        df = pd.DataFrame(self.inductive_results)
        df["inference_protocol"] = "inductive_fixed_holdout"
        self.inductive_results_df = df
        self.inductive_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        self.inductive_gates_df = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
        df.to_csv(os.path.join(self.out_dir, "inductive_fold_metrics.csv"), index=False)
        self.inductive_predictions_df.to_csv(os.path.join(self.out_dir, "inductive_predictions.csv"), index=False)
        self.inductive_gates_df.to_csv(os.path.join(self.out_dir, "inductive_drug_gates.csv"), index=False)
        print("\n================ Done Inductive Evaluation ================")
        print("Shape:", df.shape)
        return df



def load_cv_fit(path, map_location="cpu"):
    """Load one `fit_<rep>_<fold>.pt` bundle written by `run_experiments(checkpoint_dir=...)`.
    The bundle pickles full modules, so this is for trusted local files only."""
    bundle = torch.load(path, map_location=map_location, weights_only=False)
    if bundle.get("protocol") != "cv" or bundle.get("format_version") != 1:
        raise ValueError(f"{path} is not a version-1 CV fit bundle")
    return bundle


def score_saved_cv_fit(bundle, pairs=None, device="cpu"):
    """Score `pairs` (default: the bundle's own test fold) with every saved model,
    using exactly the transforms of the original run. Returns a long frame with
    columns model, drug1, drug2, label, score, threshold; on the saved test fold
    the scores reproduce `oof_predictions.csv` for that (repetition, fold)."""
    from dtpkg.metrics import evaluate_by_pair_bins_fusion, evaluate_baseline_inductive, evaluate_pair_scores
    pairs = bundle["test_pairs"] if pairs is None else pairs
    emb_df, topo_df, cats = bundle["embeddings"], bundle["eval_topology"], bundle["drug_to_cat"]
    frames = []
    for name, model in bundle["models"].items():
        model = deepcopy(model).to(device).eval()
        if name == "baseline":
            _, scored = evaluate_baseline_inductive(model, pairs, emb_df, cats, device,
                                                    scaler=bundle["baseline_scaler"], return_predictions=True)
        elif name == "topo_only":
            mu, sigma, cols = bundle["topo_only_normalization"]
            with torch.no_grad():
                def feature(drug):
                    a = (topo_df.loc[str(drug), cols].to_numpy(float) - mu) / sigma
                    return torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                scores = [torch.sigmoid(model(feature(r.drug1), feature(r.drug2))).item()
                          for r in pairs.itertuples(index=False)]
            _, scored = evaluate_pair_scores(pairs, scores, cats)
        else:  # fusion variants share the LatentGateModel interface
            _, scored = evaluate_by_pair_bins_fusion(model, pairs, emb_df, topo_df, cats,
                                                     device=device, return_predictions=True)
        scored = scored.rename(columns={"pred": "score"})
        if "threshold" not in scored:
            scored["threshold"] = .5
        frames.append(scored[["drug1", "drug2", "label", "score", "threshold"]].assign(model=name))
    return pd.concat(frames, ignore_index=True)
