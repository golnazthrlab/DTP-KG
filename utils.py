from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import ttest_rel


# --- Canonical pair handling ---
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
        "prec": precision_score(y_true, y_hat, zero_division=0),
        "rec": recall_score(y_true, y_hat, zero_division=0),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "auc": auc,
    }

def evaluate_by_pair_bins(model, loader, pairs_df, drug_to_cat,
                          device=None, min_samples=0):
    """
    Baseline eval (MeSH-only). Returns dict[bin_name] -> metric dict.
    min_samples=0 so we don't silently drop bins.
    """
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
    for key in PAIR_BINS:
        sub = df[df["pair_bin"] == key]
        if len(sub) < min_samples:
            continue
        y_true = sub["label"].astype(int).to_numpy()
        y_prob = sub["pred"].to_numpy()
        y_hat  = sub["binary"].astype(int).to_numpy()
        results[key] = _bin_result_dict(y_true, y_prob, y_hat)
    return results


def evaluate_by_pair_bins_fusion(model, pairs_df, emb_df, topo_df, drug_to_cat,
                                 device=None, min_samples=0):
    """
    Fusion eval (needs per-drug bio+topo lookups).
    Returns dict[bin_name] -> metric dict.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    emb_dict  = emb_df.to_dict(orient="index")
    topo_dict = topo_df.to_dict(orient="index")
    emb_cols  = list(emb_df.columns)
    topo_cols = list(topo_df.columns)
    bio_dim   = emb_df.shape[1]
    topo_dim  = topo_df.shape[1]

    def get_feat(drug, src, cols, dim):
        if drug not in src:
            return torch.zeros(dim, dtype=torch.float32)
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
    for key in PAIR_BINS:
        sub = df[df["pair_bin"] == key]
        if len(sub) < min_samples:
            continue
        y_true = sub["label"].astype(int).to_numpy()
        y_prob = sub["pred"].to_numpy()
        y_hat  = sub["binary"].astype(int).to_numpy()
        results[key] = _bin_result_dict(y_true, y_prob, y_hat)
    return results



def compare_models(df, metrics=["acc","prec","rec","f1","auc"]):
    comparisons = []
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        for metric in metrics:
            base_vals = sub[sub["model"] == "baseline"][metric].values
            fus_vals  = sub[sub["model"] == "fusion"][metric].values
            if len(base_vals) == len(fus_vals) and len(base_vals) > 1:
                t, p = ttest_rel(base_vals, fus_vals)
                comparisons.append({"category": cat, "metric": metric, "t": t, "p": p})
    return pd.DataFrame(comparisons)



def plot_performance_by_category_with_sig(results_df,
                                          out_dir="results/plots",
                                          metrics=("acc","prec","rec","f1","auc"),
                                          model_labels=("baseline","fusion"),
                                          colors=("tab:blue","tab:orange"),
                                          show=True):
    """
    For each category:
      - plots mean ± std per metric for baseline & fusion
      - adds red '*' above fusion bar if p<0.05 and fusion > baseline
    """
    os.makedirs(out_dir, exist_ok=True)

    categories = sorted(results_df["category"].unique())
    mA, mB = model_labels
    width = 0.35

    for cat in categories:
        df_cat = results_df[results_df["category"] == cat]

        # --- gather per-metric stats ---
        summary = {}
        for model in [mA, mB]:
            summary[model] = {}
            sub = df_cat[df_cat["model"] == model]
            for m in metrics:
                summary[model][f"{m}_mean"] = np.nanmean(sub[m])
                summary[model][f"{m}_std"]  = np.nanstd(sub[m])

        # --- significance tests ---
        sig_marks = {}
        for m in metrics:
            base = df_cat[df_cat["model"] == mA][m].values
            fus  = df_cat[df_cat["model"] == mB][m].values
            if len(base) == len(fus) and len(base) > 1:
                t, p = ttest_rel(base, fus, nan_policy="omit")
                sig_marks[m] = "*" if (p < 0.05 and np.nanmean(fus) > np.nanmean(base)) else ""
            else:
                sig_marks[m] = ""

        # --- plot ---
        x = np.arange(len(metrics))
        means_A = [summary[mA][f"{m}_mean"] for m in metrics]
        stds_A  = [summary[mA][f"{m}_std"]  for m in metrics]
        means_B = [summary[mB][f"{m}_mean"] for m in metrics]
        stds_B  = [summary[mB][f"{m}_std"]  for m in metrics]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(x - width/2, means_A, width, yerr=stds_A,
               label=mA, color=colors[0], capsize=4)
        ax.bar(x + width/2, means_B, width, yerr=stds_B,
               label=mB, color=colors[1], capsize=4)

        # --- add significance stars above fusion bars ---
        for i, metric in enumerate(metrics):
            if sig_marks[metric]:
                ax.text(x[i] + width/2, means_B[i] + stds_B[i] + 0.015,
                        sig_marks[metric], ha="center", va="bottom",
                        fontsize=16, color="red", fontweight="bold")

        # --- styling ---
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05 * max(means_A + means_B))
        ax.set_title(f"Performance by Metric — {cat}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        save_path = os.path.join(out_dir, f"{cat}_performance_bars.png")
        fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    print(f"Plots (mean ± std, significance) saved and shown for {len(categories)} categories → {out_dir}")



def plot_metric_bars_3models_by_category(summary_dict, sig_tests,
                                         model_labels=("baseline","fusion","fusion_with_go"),
                                         metrics=("AUC","AP","F1","ACC"),
                                         colors=("tab:blue","tab:orange","tab:green"),
                                         figsize=(14, 10),
                                         title="Model Comparison by Category"):
    """
    Draw one subplot per category, showing 3-model bars for each metric.
    Includes paired significance markers (*).
    """

    categories = list(summary_dict.keys())
    n_cats = len(categories)
    n_rows = int(np.ceil(n_cats / 3))
    n_cols = min(3, n_cats)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    for ax, cat in zip(axes, categories):
        m1, m2, m3 = model_labels
        x = np.arange(len(metrics))
        width = 0.25

        means = {m: [summary_dict[cat][m][f"{met}_mean"] for met in metrics] for m in model_labels}
        stds  = {m: [summary_dict[cat][m][f"{met}_std"]  for met in metrics] for m in model_labels}

        ax.bar(x - width, means[m1], width, yerr=stds[m1], label=m1, color=colors[0], capsize=3)
        ax.bar(x,         means[m2], width, yerr=stds[m2], label=m2, color=colors[1], capsize=3)
        ax.bar(x + width, means[m3], width, yerr=stds[m3], label=m3, color=colors[2], capsize=3)

        # --- significance marks ---
        for i, met in enumerate(metrics):
            y_max = max(means[m1][i]+stds[m1][i], means[m2][i]+stds[m2][i], means[m3][i]+stds[m3][i])
            y_sig = y_max + 0.03
            p_ab = sig_tests[cat][met]["MeSH_vs_Topo"]
            p_bc = sig_tests[cat][met]["Topo_vs_TopoGO"]
            if p_ab < 0.05:
                ax.plot([x[i]-width, x[i]], [y_sig, y_sig], color="k", lw=1)
                ax.text(x[i]-width/2, y_sig+0.01, "*", ha="center", va="bottom", fontsize=11)
            if p_bc < 0.05:
                ax.plot([x[i], x[i]+width], [y_sig, y_sig], color="k", lw=1)
                ax.text(x[i]+width/2, y_sig+0.01, "*", ha="center", va="bottom", fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=0)
        ax.set_title(cat, fontsize=11)
        ax.set_ylim(0, 1.05)

    # Legend & title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()  


def summarize_results_for_plot_by_category(df, metrics=("AUC","AP","F1","ACC")):
    """
    Summarize performance metrics across models, folds, and categories.
    Returns:
        summary_dict[category][model][metric_mean/std]
        sig_tests[category][metric][comparison]
    """
    summary_dict = {}
    sig_tests = {}

    for cat, subcat in df.groupby("CATEGORY"):
        summary_dict[cat] = {}
        sig_tests[cat] = {}

        # compute mean/std per model
        for model_name, sub in subcat.groupby("MODEL"):
            summary_dict[cat][model_name] = {}
            for m in metrics:
                summary_dict[cat][model_name][f"{m}_mean"] = sub[m].mean()
                summary_dict[cat][model_name][f"{m}_std"]  = sub[m].std(ddof=1)

        # paired significance test per metric (baseline↔fusion, fusion↔fusion+GO)
        folds = df["FOLD"].unique()
        models = df["MODEL"].unique()

        if all(x in models for x in ["BASELINE", "FUSION", "FUSION_WITH_GO"]):
            for m in metrics:
                baseline_vals = [subcat[(subcat.MODEL=="BASELINE") & (subcat.FOLD==f)][m].mean() for f in folds]
                fusion_vals   = [subcat[(subcat.MODEL=="FUSION") & (subcat.FOLD==f)][m].mean() for f in folds]
                fusion_go_vals= [subcat[(subcat.MODEL=="FUSION_WITH_GO") & (subcat.FOLD==f)][m].mean() for f in folds]

                _, p_ab = ttest_rel(baseline_vals, fusion_vals)
                _, p_bc = ttest_rel(fusion_vals, fusion_go_vals)
                sig_tests[cat][m] = {"MeSH_vs_Topo": p_ab, "Topo_vs_TopoGO": p_bc}

    print(f"✅ Computed per-category summaries for {len(summary_dict)} categories.")
    return summary_dict, sig_tests

def plot_inductive_performance_with_sig(
        results_df,
        out_dir="results/inductive_plots",
        metrics=("acc","prec","rec","f1","auc"),
        model_labels=("baseline","fusion"),
        colors=("tab:blue","tab:orange"),
        show=True):
    """
    Plot inductive evaluation performance by interaction category.
    
    For each category (LL, LM, ...):
        - compute mean ± std per metric for baseline & fusion
        - perform paired t-test over repeated experiments/folds
        - mark significant improvements with red '*'
        - save individual bar plots per category
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel

    os.makedirs(out_dir, exist_ok=True)

    categories = sorted(results_df["category"].unique())
    mA, mB = model_labels
    width = 0.35

    for cat in categories:
        df_cat = results_df[results_df["category"] == cat]

        # --- gather per-metric means/stds ---
        summary = {}
        for model in [mA, mB]:
            summary[model] = {}
            sub = df_cat[df_cat["model"] == model]
            for m in metrics:
                summary[model][f"{m}_mean"] = np.nanmean(sub[m])
                summary[model][f"{m}_std"]  = np.nanstd(sub[m])

        # --- significance tests ---
        sig_marks = {}
        for m in metrics:
            base_vals = df_cat[df_cat["model"] == mA][m].values
            fus_vals  = df_cat[df_cat["model"] == mB][m].values

            if len(base_vals) == len(fus_vals) and len(base_vals) > 1:
                t, p = ttest_rel(base_vals, fus_vals, nan_policy="omit")
                sig_marks[m] = "*" if (p < 0.05 and np.nanmean(fus_vals) > np.nanmean(base_vals)) else ""
            else:
                sig_marks[m] = ""

        # --- bar plots ---
        x = np.arange(len(metrics))
        means_A = [summary[mA][f"{m}_mean"] for m in metrics]
        stds_A  = [summary[mA][f"{m}_std"]  for m in metrics]
        means_B = [summary[mB][f"{m}_mean"] for m in metrics]
        stds_B  = [summary[mB][f"{m}_std"]  for m in metrics]

        fig, ax = plt.subplots(figsize=(6,4))

        ax.bar(x - width/2, means_A, width, yerr=stds_A,
               label=mA, color=colors[0], capsize=4)
        ax.bar(x + width/2, means_B, width, yerr=stds_B,
               label=mB, color=colors[1], capsize=4)

        # --- add significance stars ---
        for i, metric in enumerate(metrics):
            if sig_marks[metric]:
                ax.text(x[i] + width/2,
                        means_B[i] + stds_B[i] + 0.015,
                        sig_marks[metric],
                        ha="center", va="bottom",
                        fontsize=16, color="red", fontweight="bold")

        # --- styling ---
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05 * max(means_A + means_B))
        ax.set_title(f"Inductive Performance — {cat}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        save_path = os.path.join(out_dir, f"{cat}_inductive_bars.png")
        fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    print(f"[Inductive] Plots saved for {len(categories)} categories → {out_dir}")

def evaluate_baseline_inductive(model, pairs_df, emb_df, drug_to_cat, device):
    model.eval()
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
            x = np.concatenate(vec1 + vec2)

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().item()
    

            preds.append(prob)
            labels.append(y)

            # Added for sanity check - might delete later
            print(preds[:100])
            print(np.mean(preds), np.min(preds), np.max(preds))

    df = pairs_df.copy()
    df["pred"] = preds
    df["binary"] = (df["pred"] >= 0.5).astype(int)
    df["cat1"] = df["drug1"].map(drug_to_cat)
    df["cat2"] = df["drug2"].map(drug_to_cat)
    df["pair_bin"] = df.apply(lambda r: _canon_pair(r["cat1"], r["cat2"]), axis=1)

    # Aggregation same as before
    out = {}
    for key in PAIR_BINS:
        sub = df[df["pair_bin"] == key]
        if len(sub) == 0:
            continue

        y_true = sub["label"].astype(int).to_numpy()
        y_prob = sub["pred"].to_numpy()
        y_hat  = sub["binary"].astype(int).to_numpy()
        out[key] = _bin_result_dict(y_true, y_prob, y_hat)

    return out



def plot_baseline_cv_vs_inductive(results_df):
    """
    Plot baseline CV vs Inductive results for:
    ACC, Precision, Recall, F1, AUC.
    """

    metrics = ["acc", "precision", "recall", "f1", "auc"]
    cv_means = []
    ind_means = []

    # compute mean across experiments (and across categories)
    for m in metrics:
        cv_means.append(results_df[results_df.eval_type == "cv"][m].mean())
        ind_means.append(results_df[results_df.eval_type == "inductive"][m].mean())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, cv_means, width, label="CV Performance")
    bars2 = ax.bar(x + width/2, ind_means, width, label="Inductive Performance")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model: CV vs Inductive Performance")
    ax.legend()

    plt.tight_layout()
    plt.show()