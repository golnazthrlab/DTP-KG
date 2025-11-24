import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
import pandas as pd

from MeSHDataLoaders import DDIDataset, get_dataloaders
from utils import evaluate_by_pair_bins, evaluate_by_pair_bins_fusion, evaluate_baseline_inductive

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
        patience=5,
        out_dir="results",
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
        self.patience = patience
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # --- Initialize result containers dynamically ---
        self.results = {"baseline": [], "fusion": []}
        if self.topo_extractor_with_go is not None and self.fusion_model_go is not None:
            self.results["fusion_with_go"] = []

        print("FusionTrainer initialized.")
        print(f"   → Using GO extractor: {self.topo_extractor_with_go is not None}")

    def _should_stop(self, val_losses,  min_delta=1e-4):
        if len(val_losses) <= self.patience:
            return False
        recent = val_losses[-self.patience:]
        best = min(val_losses)
        # stop if no improvement in last `patience` epochs
        return all(l > best - min_delta for l in recent)   
    
    def _evaluate_bins_topo_only(self, model, topo_df, test_pairs, mu, sigma, topo_cols):
        device = self.device

        # --- normalization vectors ---
        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
        sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

        # --- normalized topo getter ---
        def get_norm_topo(drug):
            raw = torch.tensor(
                [topo_df.loc[str(drug), c] for c in topo_cols],
                dtype=torch.float32,
                device=device,
            )
            return (raw - mu_t) / sigma_t

        # --- storage ---
        preds = []
        trues = []
        cats = []

        # --- evaluate ---
        model.eval()
        with torch.no_grad():
            for _, row in test_pairs.iterrows():
                d1, d2, y = row["drug1"], row["drug2"], row["label"]

                t1 = get_norm_topo(d1).unsqueeze(0)
                t2 = get_norm_topo(d2).unsqueeze(0)

                logit = model(t1, t2).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()

                preds.append(prob)
                trues.append(y)

                # drug categories from trainer (bio-independent)
                c1 = self.drug_to_cat[str(d1)]
                c2 = self.drug_to_cat[str(d2)]

                key = f"{c1}_{c2}" if c1 <= c2 else f"{c2}_{c1}"
                cats.append(key)

        preds = np.array(preds)
        trues = np.array(trues)
        cats = np.array(cats)

        # --- prepare output ---
        unique_bins = sorted(set(cats))
        out = {}

        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

        for b in unique_bins:
            idx = np.where(cats == b)[0]
            if len(idx) == 0:
                continue

            y_true = trues[idx]
            y_pred = preds[idx]
            y_bin  = (y_pred >= 0.5).astype(int)

            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = float("nan")

            out[b] = {
                "auc":         float(auc),
                "acc":         float(accuracy_score(y_true, y_bin)),
                "f1":          float(f1_score(y_true, y_bin, zero_division=0)),
                "precision":   float(precision_score(y_true, y_bin, zero_division=0)),
                "recall":      float(recall_score(y_true, y_bin, zero_division=0)),
            }

        return out   
        
       

    def _split_train_val(self, X, y, pairs, val_split):
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset, DataLoader

        # --- split indexes ---
        idx_train, idx_val = train_test_split(
            np.arange(len(y)),
            test_size=val_split,
            random_state=self.seed,
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

    def _measure_gates_on_pairs(self, model, pairs_df, topo_df, emb_df):
        model.eval()

        topo_dict = topo_df.to_dict(orient="index")
        emb_dict = emb_df.to_dict(orient="index")

        bio_dim = emb_df.shape[1]
        topo_dim = topo_df.shape[1]
        emb_cols = list(emb_df.columns)
        topo_cols = list(topo_df.columns)

        def get_feat(drug, src, dim, cols):
            if drug not in src:
                return torch.zeros(dim)
            return torch.tensor([src[drug][c] for c in cols], dtype=torch.float32)

        gate_vals = []

        with torch.no_grad():
            for _, row in pairs_df.iterrows():
                d1, d2 = row["drug1"], row["drug2"]

                bio1 = get_feat(d1, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                bio2 = get_feat(d2, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                topo1 = get_feat(d1, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                topo2 = get_feat(d2, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)

                _, (g1, g2) = model(bio1, topo1, bio2, topo2)
                topo_w = float((g1.mean() + g2.mean()) / 2)
                gate_vals.append(topo_w)

        arr = np.array(gate_vals)
        return float(arr.mean()), float(arr.std())    
    
    def _train_fusion(
        self,
        model,
        train_pairs, test_pairs,
        topo_df, emb_df, drug_to_cat,
        patience=5, max_epochs=100,
        return_gate=False
    ):
        model = deepcopy(model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # ---- lookup dicts ----
        topo_dict = topo_df.to_dict(orient="index")
        emb_dict = emb_df.to_dict(orient="index")

        bio_dim = emb_df.shape[1]
        topo_dim = topo_df.shape[1]
        emb_cols = list(emb_df.columns)
        topo_cols = list(topo_df.columns)

        def get_feat(drug, src, dim, cols):
            if drug not in src:
                return torch.zeros(dim)
            return torch.tensor([src[drug][c] for c in cols], dtype=torch.float32)

        # tracking
        val_losses = []
        best_loss = float("inf")
        best_state = None
        best_gate_vals = None  # store gates for BEST epoch

        # ============================
        # TRAIN LOOP
        # ============================
        for epoch in range(max_epochs):
            model.train()
            epoch_losses = []

            # ---- train ----
            for _, row in train_pairs.iterrows():
                d1, d2, y = row["drug1"], row["drug2"], row["label"]

                bio1 = get_feat(d1, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                bio2 = get_feat(d2, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                topo1 = get_feat(d1, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                topo2 = get_feat(d2, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                yb = torch.tensor([y], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                logits, _ = model(bio1, topo1, bio2, topo2)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # ============================
            # VALIDATION
            # ============================
            model.eval()
            val_losses_epoch = []
            gate_vals_epoch = []  # gates for THIS epoch

            with torch.no_grad():
                for _, row in test_pairs.iterrows():
                    d1, d2, y = row["drug1"], row["drug2"], row["label"]

                    bio1 = get_feat(d1, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                    bio2 = get_feat(d2, emb_dict, bio_dim, emb_cols).unsqueeze(0).to(self.device)
                    topo1 = get_feat(d1, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                    topo2 = get_feat(d2, topo_dict, topo_dim, topo_cols).unsqueeze(0).to(self.device)
                    yb = torch.tensor([y], dtype=torch.float32).to(self.device)

                    logits, (_, _) = model(bio1, topo1, bio2, topo2)
                    val_losses_epoch.append(criterion(logits, yb).item())

            mean_val_loss = np.mean(val_losses_epoch)
            val_losses.append(mean_val_loss)

            # If this is the best epoch, store its gate values
            if mean_val_loss < best_loss - 1e-4:
                best_loss = mean_val_loss
                best_state = deepcopy(model.state_dict())
                best_gate_vals = gate_vals_epoch.copy()

            if self._should_stop(val_losses):
                break

        # RESTORE BEST MODEL
        if best_state is not None:
            model.load_state_dict(best_state)

        # Get the topo weights 
        mean_topo, std_topo = self._measure_gates_on_pairs(
            model=model,
            pairs_df=test_pairs,
            topo_df=topo_df,
            emb_df=emb_df
        )

        gate_vals_summary = {
            "mean_topo": mean_topo,
            "std_topo": std_topo,
        }

        # Evaluate model
        bin_results = evaluate_by_pair_bins_fusion(
            model=model,
            pairs_df=test_pairs,
            emb_df=emb_df,
            topo_df=topo_df,
            drug_to_cat=drug_to_cat,
            device=self.device,
        )

        if return_gate:
            return bin_results, gate_vals_summary
        else:
            return bin_results


    # TRAIN BASELINE (MeSH-only classifier)
    def _train_baseline(
        self, train_loader, test_loader, test_pairs, drug_to_cat,
       max_epochs=100
    ):
        model = deepcopy(self.baseline_cls).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        val_losses, best_loss, best_state = [], float("inf"), None

        for epoch in range(max_epochs):
            # ---- Train ----
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(-1), yb)
                loss.backward()
                optimizer.step()

            # ---- Validate ----
            model.eval()
            losses = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss = criterion(model(xb).squeeze(-1), yb).item()
                    losses.append(val_loss)
            mean_val_loss = np.mean(losses)
            val_losses.append(mean_val_loss)

            if mean_val_loss < best_loss - 1e-4:
                best_loss = mean_val_loss
                best_state = deepcopy(model.state_dict())

            if self._should_stop(val_losses):
                print(f"Early stop at epoch {epoch+1} (val loss = {mean_val_loss:.4f})")
                break

        model.load_state_dict(best_state)

        # ---- Evaluate by pair bins ----
        bin_results = evaluate_by_pair_bins(model, test_loader, test_pairs, drug_to_cat)
        print("Baseline evaluation:", {k: v["auc"] for k, v in bin_results.items()})
        return bin_results

    def run_experiments(self, X, y, pairs, folds, emb_df, drug_to_cat, num_experiments=3):
        """
        Run experiments for:
        - Baseline (MeSH only)
        - Fusion (MeSH + Topo)
        - Fusion_with_GO (MeSH + Topo + GO)  optional, only if topo_extractor_with_go is provided
        """
        self.results = {}

        for exp in range(num_experiments):
            print(f"\n=== Experiment {exp+1}/{num_experiments} ===")
            self.results[exp] = {}

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                print(f"\n--- Fold {fold_idx+1}/{len(folds)} ---")

                # === Split data ===
                tr_loader, te_loader, tr_idx, te_idx = get_dataloaders(
                    X, y, folds, fold_idx, batch_size=self.batch_size
                )
                train_pairs = pairs.iloc[tr_idx].reset_index(drop=True)
                test_pairs = pairs.iloc[te_idx].reset_index(drop=True)

                all_drugs = list(
                    set(train_pairs["drug1"])
                    .union(train_pairs["drug2"])
                    .union(test_pairs["drug1"])
                    .union(test_pairs["drug2"])
                )

                # === Compute topological features (no GO) ===
                topo_df = self.topo_extractor.compute_for_fold(all_drugs, eval_pairs=test_pairs)
                topo_df = topo_df.reindex(all_drugs).fillna(0)

                # === Compute topological features (with GO), if extractor provided ===
                topo_df_with_go = None
                if hasattr(self, "topo_extractor_with_go") and self.topo_extractor_with_go is not None:
                    topo_df_with_go = self.topo_extractor_with_go.compute_for_fold(all_drugs, eval_pairs=test_pairs)
                    topo_df_with_go = topo_df_with_go.reindex(all_drugs).fillna(0)

                # === Train models ===
                print("\n→ Training baseline model (MeSH only)")
                baseline_bins = self._train_baseline(tr_loader, te_loader, test_pairs, drug_to_cat)

                print("\n→ Training fusion model (MeSH + Topo)")
                fusion_bins_wo_go = self._train_fusion(self.fusion_model_wo_go, train_pairs, test_pairs, topo_df, emb_df, drug_to_cat)

                # Optional 3rd model
                if topo_df_with_go is not None:
                    print("\n→ Training fusion_with_GO model (MeSH + Topo + GO)")
                    fusion_go_bins = self._train_fusion(self.fusion_model_go, train_pairs, test_pairs, topo_df_with_go, emb_df, drug_to_cat)
                else:
                    fusion_go_bins = None

                # === Store results ===
                fold_results = {}
                all_bins = set(baseline_bins.keys()) | set(fusion_bins_wo_go.keys())
                if fusion_go_bins is not None:
                    all_bins |= set(fusion_go_bins.keys())

                for bin_name in all_bins:
                    fold_results[bin_name] = {
                        "baseline": baseline_bins.get(bin_name, {}),
                        "fusion": fusion_bins_wo_go.get(bin_name, {}),
                    }
                    if fusion_go_bins is not None:
                        fold_results[bin_name]["fusion_with_go"] = fusion_go_bins.get(bin_name, {})

                self.results[exp][fold_idx] = fold_results

        # === Save results ===
        self._save_detailed_results()
        print("\n✅ All experiments completed and saved.")




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
                        rows.append(row)

        df = pd.DataFrame(rows)
        self.results_df = df  # attach for later analysis

        if filename is None:
            os.makedirs(self.out_dir, exist_ok=True)
            filename = os.path.join(self.out_dir, "detailed_results.pkl")

        with open(filename, "wb") as f:
            pickle.dump(self.results, f)

        print(f"Saved detailed results to {filename}")
        print(f"DataFrame shape: {df.shape}")
        return df        


    def run_inductive_eval(
        self,
        X, y, pairs,
        emb_df,
        topo_inductive_df,
        inductive_test_pairs,
        drug_to_cat,
        num_experiments=5,
        val_split=0.15,
    ):

        print("\n========================================================")
        print("               Running Inductive Evaluation             ")

        self.inductive_results = []

        for exp in range(num_experiments):
            print(f"\n========== Inductive Experiment {exp+1}/{num_experiments} ==========\n")

            # STEP 1 — Train/Validation split
            train_loader, val_loader, train_pairs_df, val_pairs_df = \
                self._split_train_val(X, y, pairs, val_split)

            # STEP 2 — Compute train/val topo features
            # Identify all drugs used in train or val
            trainval_drugs = list(
                set(train_pairs_df["drug1"]).union(train_pairs_df["drug2"])
                .union(val_pairs_df["drug1"]).union(val_pairs_df["drug2"])
            )

            # Leakage-safe removal of val DDIs
            topo_trainval_df = self.topo_extractor.compute_for_fold(
                drug_ids=trainval_drugs,
                eval_pairs=val_pairs_df  # remove edges of val fold
            )
            topo_trainval_df = topo_trainval_df.reindex(trainval_drugs).fillna(0)

            # --------------------------------------------------
            # STEP 3 — Train baseline
            # --------------------------------------------------
            print("Training Baseline (MeSH only)")
            _ = self._train_baseline(
                train_loader=train_loader,
                test_loader=val_loader,
                test_pairs=val_pairs_df,
                drug_to_cat=drug_to_cat,
                max_epochs=self.epochs
            )

            # --------------------------------------------------
            # STEP 4 — Train fusion model
            # --------------------------------------------------
            print("\nTraining Fusion Model (MeSH + Topo)")
            _ = self._train_fusion(
                model=self.fusion_model_wo_go,
                train_pairs=train_pairs_df,
                test_pairs=val_pairs_df,
                topo_df=topo_trainval_df,
                emb_df=emb_df,
                drug_to_cat=drug_to_cat
            )
            # --------------------------------------------------
            # 5. Inductive evaluation
            # --------------------------------------------------
            print("\nEvaluating on inductive test set...")

            # baseline
            baseline_inductive = evaluate_baseline_inductive(
                model=self.baseline_cls.to(self.device),
                pairs_df=inductive_test_pairs,
                emb_df=emb_df,
                drug_to_cat=drug_to_cat,
                device=self.device
            )

            # fusion
            fusion_inductive = evaluate_by_pair_bins_fusion(
                model=self.fusion_model_wo_go.to(self.device),
                pairs_df=inductive_test_pairs,
                emb_df=emb_df,
                topo_df=topo_inductive_df,
                drug_to_cat=drug_to_cat,
                device=self.device
            )

            # store
            for bin_name in fusion_inductive.keys():
                self.inductive_results.append({
                    "experiment": exp,
                    "category": bin_name,
                    "model": "baseline",
                    **baseline_inductive[bin_name]
                })
                self.inductive_results.append({
                    "experiment": exp,
                    "category": bin_name,
                    "model": "fusion",
                    **fusion_inductive[bin_name]
                })

        # produce final DF
        df = pd.DataFrame(self.inductive_results)
        self.inductive_results_df = df
        print("\n================ Done Inductive Evaluation ================")
        print("Shape:", df.shape)
        return df
    
    def run_inductive_baseline_eval(
        self,
        X, y, pairs,
        emb_df,
        inductive_test_pairs,
        drug_to_cat,
        num_experiments=5,
        val_split=0.15
    ):
        """
        Run inductive evaluation using ONLY the baseline (MeSH-only) classifier.
        Collects both CV results (from validation set) and IND results.
        """

        print("\n========================================================")
        print("         Running BASELINE-ONLY Inductive Eval           ")
        print("========================================================")

        results = []

        for exp in range(num_experiments):
            print(f"\n========== Baseline Inductive Experiment {exp+1}/{num_experiments} ==========\n")

            # STEP 1 — train/val split
            train_loader, val_loader, train_pairs_df, val_pairs_df = \
                self._split_train_val(X, y, pairs, val_split)

            # STEP 2 — train baseline (returns CV metrics per pair-bin)
            print("Training baseline (MeSH only)...")
            baseline_cv = self._train_baseline(
                train_loader=train_loader,
                test_loader=val_loader,
                test_pairs=val_pairs_df,
                drug_to_cat=drug_to_cat,
                max_epochs=self.epochs
            )

            # For each BIN store CV metrics
            for bin_name, bin_stats in baseline_cv.items():
                results.append({
                    "experiment": exp,
                    "eval_type": "cv",
                    "category": bin_name,
                    "auc":        bin_stats.get("auc", None),
                    "acc":        bin_stats.get("acc", None),
                    "precision":  bin_stats.get("prec", None),
                    "recall":     bin_stats.get("rec", None),
                    "f1":         bin_stats.get("f1", None)
                })

            # STEP 3 — evaluate baseline on inductive test pairs
            print("\nEvaluating BASELINE on inductive test set...")

            baseline_inductive = evaluate_baseline_inductive(
                model=self.baseline_cls.to(self.device),
                pairs_df=inductive_test_pairs,
                emb_df=emb_df,
                drug_to_cat=drug_to_cat,
                device=self.device
            )

            # Store inductive results
            for bin_name, bin_stats in baseline_inductive.items():
                results.append({
                    "experiment": exp,
                    "eval_type": "inductive",
                    "category": bin_name,
                    "auc":        bin_stats.get("auc", None),
                    "acc":        bin_stats.get("acc", None),
                    "precision":  bin_stats.get("prec", None),
                    "recall":     bin_stats.get("rec", None),
                    "f1":         bin_stats.get("f1", None)
                })

        df = pd.DataFrame(results)
        print("\n================ Done Baseline-Only Inductive Eval ================")
        print("Shape:", df.shape)

        self.baseline_inductive_results_df = df
        return df