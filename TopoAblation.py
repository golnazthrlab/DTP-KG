import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from MeSHDataLoaders import get_dataloaders
from LatentGateModel import LatentGateModel

class TopoOnlyModel(nn.Module):
    def __init__(self, topo_dim, hidden=(64, 32), dropout=0.1):
        super().__init__()

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

    def forward(self, topo1, topo2):
        x = torch.cat([topo1, topo2], dim=1)
        # x = self.input_norm(x)   # normalize before MLP
        return self.net(x).view(-1)



feature_groups = {
    "degree": [
        "deg_total", "deg_drug", "deg_prot",
        "deg_min", "deg_max", "deg_mean", "deg_std"
    ],
    "cluster": [
        "clust_local", "boundary_ratio"
    ],
    "centrality": [
        "close_local", "katz_local", "betw_local"
    ]
}


class TopoAblation:
    """
    Perform grouped ablation experiments on topological feature subsets.
    - Removes topo columns instead of zeroing them
    - Initializes a correct LatentGateModel per mode (per topo_dim)
    """

    def __init__(
        self,
        trainer,
        topo_extractor,
        feature_groups=feature_groups,
        out_dir="results/ablation"
    ):
        self.trainer = trainer
        self.topo_extractor = topo_extractor
        self.feature_groups = feature_groups
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

        # Full topo feature count
        self.full_topo_dim = topo_extractor.num_of_topo_feats

        # Results: exp -> fold -> ablation_name -> bin_name -> metrics
        self.results = {}

    # Remove the ablated feature columns entirely
    def _apply_ablation(self, topo_df, drop_features):

        # Filter only existing columns
        missing = [c for c in drop_features if c not in topo_df.columns]
        if missing:
            print(f"[WARNING] These features do not exist in topo_df and will be skipped: {missing}")

        drop_features = [c for c in drop_features if c in topo_df.columns]

        # Drop valid columns
        topo_df = topo_df.drop(columns=drop_features)

        # Safety
        if topo_df.shape[1] == 0:
            raise ValueError(
                f"[ERROR] Ablation removed ALL topo features. "
                f"Remaining columns: {list(topo_df.columns)}"
            )

        return topo_df

    # Build fresh fusion model for a given topo_dim
    def _make_model(self, topo_dim):
        return LatentGateModel(
            bio_dim=128,
            topo_dim=topo_dim,
            gate_scalar=False,  # IMPORTANT FOR SENSITIVE ABLATION
        )
    
    def _evaluate_bins_topo_only(self, model, topo_df, test_pairs):
        return self.trainer._evaluate_bins_topo_only(model, topo_df, test_pairs)

    def _save_topo_only_results(self):
        out = os.path.join(self.out_dir, "topo_only_results.pkl")
        with open(out, "wb") as f:
            pickle.dump(self.topo_only_results, f)
        print(f"Saved topo-only results → {out}") 

    def plot_topo_only_per_category(self, save_path="results/ablation/topo_only"):
        """
        Plot per-category results for topo-only models.
        We plot all metrics: auc, acc, f1, precision, recall.
        """

        import matplotlib.pyplot as plt
        import numpy as np
        import os

        os.makedirs(save_path, exist_ok=True)

        metrics = ["auc", "acc", "f1", "precision", "recall"]
        modes = ["topo_only_full", "topo_only_wo_degree",
                "topo_only_wo_cluster", "topo_only_wo_centrality"]

        results = self.topo_only_results

        # extract categories
        example_exp = list(results.keys())[0]
        example_fold = list(results[example_exp].keys())[0]
        categories = list(results[example_exp][example_fold]["topo_only_full"].keys())

        for cat in categories:
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(f"Topo-Only Ablation — Category: {cat}", fontsize=16)

            for j, metric in enumerate(metrics):
                means = []
                stds = []

                for mode in modes:
                    vals = []
                    for exp in results:
                        for fold in results[exp]:
                            vals.append(results[exp][fold][mode][cat][metric])
                    vals = np.array(vals)
                    means.append(vals.mean())
                    stds.append(vals.std())

                axs[j].bar(modes, means, yerr=stds, capsize=4)
                axs[j].set_title(metric)
                axs[j].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            fig.savefig(os.path.join(save_path, f"topo_only_{cat}.png"), dpi=150)
            plt.close(fig) 

    def plot_topo_only_summary(self, save_path=None):
        metrics = ["auc", "acc", "f1", "precision", "recall"]
        modes = ["topo_only_full", "topo_only_wo_degree",
                "topo_only_wo_cluster", "topo_only_wo_centrality"]

        if save_path is None:
            save_path = os.path.join(self.out_dir, "topo_only_summary.png")

        results = self.topo_only_results
        fig, axs = plt.subplots(1, 5, figsize=(22, 4))
        fig.suptitle("Topo-Only Ablation Summary", fontsize=16)

        for j, metric in enumerate(metrics):
            means, stds = [], []

            for mode in modes:
                vals = []

                for exp in results:
                    for fold in results[exp]:
                        for cat in results[exp][fold][mode]:
                            vals.append(results[exp][fold][mode][cat][metric])

                vals = np.array(vals)
                means.append(vals.mean())
                stds.append(vals.std())

            axs[j].bar(modes, means, yerr=stds, capsize=4)
            axs[j].set_title(metric)
            axs[j].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved topo-only summary plot → {save_path}")                     
    
    def _train_topo_only(
        self,
        model,
        train_pairs,
        test_pairs,
        topo_df,
        patience=5,
        max_epochs=20,
        lr=1e-3,
        weight_decay=1e-5,
        device="cpu",
    ):
        model = model.to(device)
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
        train_drugs = set(train_pairs["drug1"]) | set(train_pairs["drug2"])
        train_mat = topo_df.loc[list(train_drugs)].values

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
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            train_losses = []

            for _, row in train_pairs.iterrows():
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

            # ---------------------------
            # Validation
            # ---------------------------
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _, row in test_pairs.iterrows():
                    d1, d2, y = row["drug1"], row["drug2"], row["label"]

                    t1 = get_norm_topo(d1).unsqueeze(0)
                    t2 = get_norm_topo(d2).unsqueeze(0)
                    yb = torch.tensor([y], dtype=torch.float32, device=device)

                    logits = model(t1, t2)
                    val_losses.append(criterion(logits, yb).item())

            mean_val_loss = np.mean(val_losses)

            # Early stopping logic
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best state
        model.load_state_dict(best_state)

        # ---------------------------
        # Final evaluation in bins
        # ---------------------------
        eval_bins = self.trainer._evaluate_bins_topo_only(
            model, topo_df, test_pairs, mu, sigma, topo_cols
        )

        return eval_bins    
    
    def run_topo_only_ablation(
        self,
        X,
        y,
        pairs,
        folds,
        emb_df,
        drug_to_cat,
        num_experiments=1,
    ):
        print("\n=== TOPO-ONLY ABLATION EXPERIMENT ===")

        ablation_modes = {"full_topo": None}
        for name, feats in self.feature_groups.items():
            ablation_modes[f"wo_{name}"] = feats

        self.topo_only_results = {}

        for exp in range(num_experiments):
            print(f"\n=== Topo-Only Experiment {exp+1}/{num_experiments} ===")
            self.topo_only_results[exp] = {}

            for fold_idx, (_, _) in enumerate(folds):
                print(f"\n--- Fold {fold_idx+1}/{len(folds)} ---")

                _, _, tr_idx, te_idx = get_dataloaders(
                    X, y, folds, fold_idx, batch_size=128
                )

                train_pairs = pairs.iloc[tr_idx].reset_index(drop=True)
                test_pairs  = pairs.iloc[te_idx].reset_index(drop=True)

                all_drugs = list(
                    set(train_pairs["drug1"]) |
                    set(train_pairs["drug2"]) |
                    set(test_pairs["drug1"]) |
                    set(test_pairs["drug2"])
                )

                topo_df_full = self.topo_extractor.compute_for_fold(
                    drug_ids=all_drugs, eval_pairs=test_pairs
                )
                topo_df_full = topo_df_full.reindex(all_drugs).fillna(0)

                fold_dict = {}

                for mode, drop_feats in ablation_modes.items():
                    print(f"\n[Mode = {mode}]")

                    if drop_feats is None:
                        topo_df = topo_df_full
                    else:
                        topo_df = topo_df_full.drop(columns=drop_feats, errors="ignore")

                    topo_dim = topo_df.shape[1]
                    model = TopoOnlyModel(topo_dim=topo_dim)

                    eval_bins = self._train_topo_only(
                        model=model,
                        train_pairs=train_pairs,
                        test_pairs=test_pairs,
                        topo_df=topo_df,
                        patience=5,
                        max_epochs=20,
                        lr=1e-3,
                        weight_decay=1e-5,
                        device=self.trainer.device,
                    )

                    fold_dict[mode] = eval_bins

                self.topo_only_results[exp][fold_idx] = fold_dict

        self._save_topo_only_results()    

    def run_ablation(
        self,
        X,
        y,
        pairs,
        folds,
        emb_df,
        drug_to_cat,
        num_experiments=1,
    ):

        # Prepare ablation configs
        ablation_modes = {"full_topo": None}   # None 
        for group_name, drop_feats in self.feature_groups.items():
            ablation_modes[f"wo_{group_name}"] = drop_feats

        self.results = {}       # performance
        self.gate_results = {}  # gate activations

        # Experiments loop
        for exp in range(num_experiments):
            print(f"\n=== Ablation Experiment {exp+1}/{num_experiments} ===")
            self.results[exp] = {}
            self.gate_results[exp] = {}

            # Fold loop
            for fold_idx, (_, _) in enumerate(folds):
                print(f"\n--- Fold {fold_idx+1}/{len(folds)} ---")

                # Get MeSH dataloaders
                _, _, tr_idx, te_idx = get_dataloaders(
                    X, y, folds, fold_idx,
                    batch_size=self.trainer.batch_size,
                )

                train_pairs = pairs.iloc[tr_idx].reset_index(drop=True)
                test_pairs  = pairs.iloc[te_idx].reset_index(drop=True)

                # All drugs for this fold
                all_drugs = list(
                    set(train_pairs["drug1"])
                    | set(train_pairs["drug2"])
                    | set(test_pairs["drug1"])
                    | set(test_pairs["drug2"])
                )

                # Compute topo features (full)
                topo_df_full = self.topo_extractor.compute_for_fold(
                    drug_ids=all_drugs,
                    eval_pairs=test_pairs,
                )
                topo_df_full = topo_df_full.reindex(all_drugs).fillna(0)

                fold_store = {}
                fold_gate_store = {}

                # Run each ablation mode
                for mode_name, drop_feats in ablation_modes.items():
                    print(f"\n[Mode = {mode_name}]")

                    # Prepare ablated topo_df
                    if drop_feats is None:
                        topo_df = topo_df_full
                    else:
                        topo_df = self._apply_ablation(topo_df_full, drop_feats)
                        print("  Removed cols:", drop_feats)
                        print("  New topo_dim:", topo_df.shape[1])

                    # Create a fresh model with correct topo_dim
                    model = self._make_model(topo_dim=topo_df.shape[1])

                    print(f"Training {mode_name} model (topo_dim={topo_df.shape[1]})")

                    # Train + evaluate (now returns gate stats!)
                    eval_bins, gate_vals_summary = self.trainer._train_fusion(
                        model=model,
                        train_pairs=train_pairs,
                        test_pairs=test_pairs,
                        topo_df=topo_df,
                        emb_df=emb_df,
                        drug_to_cat=drug_to_cat,
                        patience=self.trainer.patience,
                        max_epochs=self.trainer.epochs,
                        return_gate=True,
                    )

                    fold_store[mode_name] = eval_bins
                    fold_gate_store[mode_name] = gate_vals_summary

                # Save results for this fold
                self.results[exp][fold_idx] = fold_store
                self.gate_results[exp][fold_idx] = fold_gate_store

        self._save_results()

    def _save_results(self):
        perf_path = os.path.join(self.out_dir, "ablation_results.pkl")
        gate_path = os.path.join(self.out_dir, "gate_results.pkl")

        with open(perf_path, "wb") as f:
            pickle.dump(self.results, f)

        with open(gate_path, "wb") as f:
            pickle.dump(self.gate_results, f)

        print(f"Saved ablation performance → {perf_path}")
        print(f"Saved ablation gate shifts → {gate_path}")


    def plot_per_category(self, save_dir=None):
        """
        Plot per-category barplots for acc, prec, rec, f1, auc.
        One figure per category.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        metrics = ["acc", "prec", "rec", "f1", "auc"]
        modes = ["full_topo", "wo_degree", "wo_cluster", "wo_centrality"]

        if save_dir is None:
            save_dir = os.path.join(self.out_dir, "plots_per_category")
        os.makedirs(save_dir, exist_ok=True)

        results = self.results

        # extract categories
        example_exp = list(results.keys())[0]
        example_fold = list(results[example_exp].keys())[0]
        categories = list(results[example_exp][example_fold]["full_topo"].keys())

        for cat in categories:
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(f"Category: {cat}", fontsize=16)

            for j, metric in enumerate(metrics):
                means, stds = [], []

                for mode in modes:
                    vals = []

                    for exp in results:
                        for fold in results[exp]:
                            if cat in results[exp][fold][mode]:
                                vals.append(results[exp][fold][mode][cat][metric])

                    vals = np.array(vals)
                    means.append(vals.mean())
                    stds.append(vals.std())

                axs[j].bar(modes, means, yerr=stds, capsize=4)
                axs[j].set_title(metric)
                axs[j].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{cat}.png"), dpi=150)
            plt.close(fig)


    def plot_summary(self, save_path=None):
        """
        Plot overall summary across all categories, folds, and experiments.
        Acc, prec, rec, f1, auc.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        metrics = ["acc", "prec", "rec", "f1", "auc"]
        modes = ["full_topo", "wo_degree", "wo_cluster", "wo_centrality"]

        if save_path is None:
            save_path = os.path.join(self.out_dir, "summary.png")

        results = self.results

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle("Overall Summary Across All Categories", fontsize=16)

        for j, metric in enumerate(metrics):
            means, stds = [], []

            for mode in modes:
                vals = []

                for exp in results:
                    for fold in results[exp]:
                        for cat in results[exp][fold][mode]:
                            vals.append(results[exp][fold][mode][cat][metric])

                vals = np.array(vals)
                means.append(vals.mean())
                stds.append(vals.std())

            axs[j].bar(modes, means, yerr=stds, capsize=4)
            axs[j].set_title(metric)
            axs[j].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)        


    def plot_gate_shift_summary(self, save_path=None):
        """
        Plot the change in average topo-gate activation across ablation modes.
        Requires self.gate_results populated during run_ablation().
        """

        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if not hasattr(self, "gate_results"):
            raise ValueError("Gate results missing. Make sure run_ablation collects gate values.")

        if save_path is None:
            save_path = os.path.join(self.out_dir, "gate_shift_summary.png")

        modes = ["full_topo", "wo_degree", "wo_cluster", "wo_centrality"]

        mean_topo = []
        std_topo = []

        # aggregate across folds + experiments
        for mode in modes:
            vals = []
            for exp in self.gate_results:
                for fold in self.gate_results[exp]:
                    vals.append(self.gate_results[exp][fold][mode]["mean_topo"])
            vals = np.array(vals)
            mean_topo.append(vals.mean())
            std_topo.append(vals.std())

        # -------------------------
        # Plot
        # -------------------------
        plt.figure(figsize=(10, 5))
        plt.bar(modes, mean_topo, yerr=std_topo, capsize=5)
        plt.ylabel("Mean Topo Gate Activation")
        plt.title("Shift in Topo Contribution Under Ablations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved gate-shift summary → {save_path}")        