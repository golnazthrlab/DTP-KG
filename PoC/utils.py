import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, accuracy_score
from math import sqrt
from scipy.stats import norm 

def seed_everything(seed=13):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def undersample_majority_idx(y, seed=13):
    rs = np.random.RandomState(seed)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    k = min(len(pos), len(neg))
    return np.r_[rs.choice(pos, k, False), rs.choice(neg, k, False)]

def eval_metrics(y, p, thr=0.5):
    yhat = (p>=thr).astype(int)
    return dict(
        AUC=roc_auc_score(y,p),
        AP=average_precision_score(y,p),
        F1=f1_score(y,yhat,zero_division=0),
        ACC=accuracy_score(y,yhat),
        PREC=precision_score(y,yhat,zero_division=0),
        REC=recall_score(y,yhat,zero_division=0)
    )

def make_repeated_splits(y, n_splits=5, n_repeats=5, base_seed=13):
    """Return a list of (tr_idx, va_idx, repeat_id)."""
    splits = []
    dummyX = np.zeros_like(y)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed+100*r)
        for tr, va in skf.split(dummyX, y):
            splits.append((tr, va, r))
    return splits


def _split_per_view(Xbio, Xtopo):
    Db = Xbio.shape[1]//2 if Xbio is not None else 0
    Dt = Xtopo.shape[1]//2 if Xtopo is not None else 0
    b1=b2=t1=t2=None
    if Xbio is not None:
        b1, b2 = Xbio[:, :Db], Xbio[:, Db:]
    if Xtopo is not None:
        t1, t2 = Xtopo[:, :Dt], Xtopo[:, Dt:]
    return (b1,b2,Db), (t1,t2,Dt)

def _fit_transform_blocks(Xb_tr, Xt_tr, Xb_va, Xt_va,
                          scale_bio=True, scale_topo=True):
    """BIO: StandardScaler; TOPO: MinMax(-1,1); fit on TRAIN union of drug1/2; apply to val."""
    (b1_tr,b2_tr,Db), (t1_tr,t2_tr,Dt) = _split_per_view(Xb_tr, Xt_tr)
    (b1_va,b2_va,_ ), (t1_va,t2_va,_ ) = _split_per_view(Xb_va, Xt_va)

    # BIO
    if Xb_tr is not None and scale_bio:
        bio_scaler = StandardScaler()
        bio_scaler.fit(np.vstack([b1_tr, b2_tr]))
        b1_tr = bio_scaler.transform(b1_tr); b2_tr = bio_scaler.transform(b2_tr)
        b1_va = bio_scaler.transform(b1_va); b2_va = bio_scaler.transform(b2_va)

    # TOPO
    if Xt_tr is not None and scale_topo:
        topo_scaler = MinMaxScaler(feature_range=(-1,1))
        topo_scaler.fit(np.vstack([t1_tr, t2_tr]))
        t1_tr = topo_scaler.transform(t1_tr); t2_tr = topo_scaler.transform(t2_tr)
        t1_va = topo_scaler.transform(t1_va); t2_va = topo_scaler.transform(t2_va)

    return (b1_tr,b2_tr,Db,b1_va,b2_va), (t1_tr,t2_tr,Dt,t1_va,t2_va)

def build_pair_features(mode,    # "concat" | "sym4"
                        b1, b2,  # per-drug BIO (or None)
                        t1, t2   # per-drug TOPO (or None)
                       ):
    """Return pair features according to mode and available views."""
    def fuse(b,t):
        if b is None: return t
        if t is None: return b
        return np.concatenate([b,t], axis=1)
    f1 = fuse(b1,t1); f2 = fuse(b2,t2)
    if mode == "concat":
        return np.concatenate([f1, f2], axis=1)
    elif mode == "sym4":
        return np.concatenate([f1, f2, np.abs(f1 - f2), f1 * f2], axis=1)
    else:
        raise ValueError("mode must be 'concat' or 'sym4'")
    

from pathlib import Path
import pandas as pd
import numpy as np
import ast

def load_dataset(folder):
    """
    Load DDI data from a directory with 4 CSV files:
        ddi_pos_train.csv
        ddi_neg_train.csv
        ddi_pos_test.csv
        ddi_neg_test.csv

    Each CSV must have:
        drug_id1, drug_feat1, drug_id2, drug_feat2
    where drug_feat* is a stringified Python list of floats.

    Returns
    -------
    X_trainval : np.ndarray  [n_train,  2 * d]
    y_trainval : np.ndarray  [n_train]
    X_test     : np.ndarray  [n_test,   2 * d]
    y_test     : np.ndarray  [n_test]
    feat_cols  : list[str]   names like f0,f1,...
    """
    folder = Path(folder)

    # read and label
    def _read_pos_neg(split):
        df_pos = pd.read_csv(folder / f"ddi_pos_{split}.csv")
        df_neg = pd.read_csv(folder / f"ddi_neg_{split}.csv")
        df_pos["label"] = 1
        df_neg["label"] = 0
        return pd.concat([df_pos, df_neg], ignore_index=True)

    df_train = _read_pos_neg("train")
    df_test  = _read_pos_neg("test")

    # quick checks
    for col in ("drug_feat1", "drug_feat2"):
        if col not in df_train.columns or col not in df_test.columns:
            raise KeyError(f"Missing expected column {col!r} in {folder}")

    # parse feature columns into arrays
    def _expand_feats(df):
        # parse lists safely
        f1 = df["drug_feat1"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        f2 = df["drug_feat2"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        mat = np.stack([np.concatenate([a,b]) for a,b in zip(f1, f2)], axis=0)
        feat_cols = [f"f{i}" for i in range(mat.shape[1])]
        return mat, feat_cols

    X_trainval, feat_cols = _expand_feats(df_train)
    X_test,      _       = _expand_feats(df_test)

    y_trainval = df_train["label"].to_numpy(dtype=np.int64)
    y_test     = df_test["label"].to_numpy(dtype=np.int64)

    # sanity: same feature dim
    if X_trainval.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train/Test feature dim mismatch: {X_trainval.shape[1]} vs {X_test.shape[1]}"
        )

    # optional warning for constant columns (can hurt training)
    const_cols = np.where(X_trainval.std(axis=0) == 0)[0]
    if len(const_cols):
        print(f"WARNING: {len(const_cols)} constant feature columns "
              f"in TRAIN (e.g., first few: {const_cols[:5].tolist()})")

    return X_trainval, y_trainval, X_test, y_test, feat_cols    
    


class MLP(nn.Module):
    def __init__(self, d, hidden=(256,128), pdrop=0.1):
        super().__init__()
        layers=[]; x=d
        for h in hidden:
            layers += [nn.Linear(x,h), nn.ReLU(), nn.Dropout(pdrop)]
            x=h
        layers += [nn.Linear(x,1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)


# === utils.py (updated training section) ======================================
def train_mlp_auc_earlystop(
    Xtr, ytr, Xva, yva,
    cfg,                # dict of hyper-params
    device=None
):

    if device is None:
        device = torch_device()

    hidden       = cfg.get("hidden", (256,128))
    pdrop        = cfg.get("pdrop", 0.1)
    epochs       = cfg.get("epochs", 20)
    bs           = cfg.get("batch_size", 256)
    lr           = cfg.get("lr", 1e-3)
    wd           = cfg.get("weight_decay", 1e-4)
    patience     = cfg.get("patience", 5)
    min_delta    = cfg.get("min_delta", 5e-4)
    seed         = cfg.get("seed", 13)

    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], hidden, pdrop).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()

    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                        torch.tensor(ytr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb)
            loss  = lossf(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32, device=device))).cpu().numpy()
            auc = roc_auc_score(yva, p)

        if auc > best_auc + min_delta:
            best_auc = auc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32, device=device))).cpu().numpy()
    return p, model

from sklearn.model_selection import StratifiedKFold
import numpy as np

def run_cv_bio_vs_biotopo(
    Xbio, Xtopo, y,
    alpha=1.0,                # scales the TOPO block: topo' = alpha * topo
    mode="concat",            # "concat" or "sym4"
    n_splits=5, n_repeats=5,
    undersample=True,
    train_cfg=None,           # dict for train_mlp_auc_earlystop
    base_seed=13
):
    """
    Repeated K-Fold CV comparing:
      - Bio-only
      - Bio + alpha * Topo   (alpha is a fixed scalar)

    Ensures paired OOF predictions for DeLong.

    Returns dict:
      {
        "per_repeat": [ {"Bio": {...}, "BioTopo": {...}}, ... ],
        "summary":    { "Bio": {...},   "BioTopo": {...} },
        "delong_ready": { "y": np.array, "bio": np.array, "biotopo": np.array }
      }
    """
    if train_cfg is None: train_cfg = {}

    per_repeat=[]
    y_all=[]; bio_oof=[]; biotopo_oof=[]

    def _avg(ms):
        keys = ms[0].keys()
        return {k: float(np.mean([d[k] for d in ms])) for k in keys}
    def _summary(per_repeat,key):
        ks = list(per_repeat[0][key].keys()); out={}
        for m in ks:
            vals = np.array([rep[key][m] for rep in per_repeat], dtype=float)
            out[f"{m}_mean"]=float(vals.mean()); out[f"{m}_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        return out

    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed+100*r)
        fold_m_bio=[]; fold_m_bt=[]
        for f,(tr,va) in enumerate(skf.split(np.zeros_like(y), y)):
            y_tr_, y_va = y[tr], y[va]
            Xb_tr_, Xb_va = Xbio[tr], Xbio[va]
            Xt_tr_, Xt_va = Xtopo[tr], Xtopo[va]

            if undersample:
                keep = undersample_majority_idx(y_tr_, seed=base_seed+r+f)
                y_tr_ = y_tr_[keep]; Xb_tr_ = Xb_tr_[keep]; Xt_tr_ = Xt_tr_[keep]

            # scale per fold with existing helper (BIO: Standard, TOPO: MinMax[-1,1])
            (b1_tr,b2_tr,Db,b1_va,b2_va), _ = _fit_transform_blocks(Xb_tr_, None, Xb_va, None)
            _, (t1_tr,t2_tr,Dt,t1_va,t2_va) = _fit_transform_blocks(None, Xt_tr_, None, Xt_va)

            # ---- Bio-only ----
            def build_pair(mode, a1,a2,c1=None,c2=None):
                if c1 is None and c2 is None:
                    f1,f2 = a1, a2
                else:
                    f1,f2 = np.concatenate([a1,c1],1), np.concatenate([a2,c2],1)
                if mode=="concat":
                    return np.concatenate([f1,f2],1)
                else:
                    return np.concatenate([f1,f2, np.abs(f1-f2), f1*f2],1)

            Xtr_bio = build_pair(mode, b1_tr, b2_tr)
            Xva_bio = build_pair(mode, b1_va, b2_va)
            cfgA = dict(train_cfg); cfgA["seed"]=train_cfg.get("seed",base_seed)+17*(r+1)+f
            p_bio,_ = train_mlp_auc_earlystop(Xtr_bio, y_tr_, Xva_bio, y_va, cfg=cfgA)
            fold_m_bio.append(eval_metrics(y_va, p_bio))

            # ---- Bio + alpha * Topo ----
            if alpha == 1.0:
                t1_tr_s, t2_tr_s = t1_tr, t2_tr
                t1_va_s, t2_va_s = t1_va, t2_va
            else:
                t1_tr_s, t2_tr_s = alpha * t1_tr, alpha * t2_tr
                t1_va_s, t2_va_s = alpha * t1_va, alpha * t2_va

            Xtr_bt = build_pair(mode, b1_tr, b2_tr, t1_tr_s, t2_tr_s)
            Xva_bt = build_pair(mode, b1_va, b2_va, t1_va_s, t2_va_s)
            cfgB = dict(train_cfg); cfgB["seed"]=train_cfg.get("seed",base_seed)+97*(r+1)+f
            p_bt,_ = train_mlp_auc_earlystop(Xtr_bt, y_tr_, Xva_bt, y_va, cfg=cfgB)
            fold_m_bt.append(eval_metrics(y_va, p_bt))

            # paired OOF collection
            y_all.append(y_va); bio_oof.append(p_bio); biotopo_oof.append(p_bt)

        per_repeat.append({"Bio":_avg(fold_m_bio), "BioTopo":_avg(fold_m_bt)})

    summary={"Bio":_summary(per_repeat,"Bio"), "BioTopo":_summary(per_repeat,"BioTopo")}
    y_pool = np.concatenate(y_all)
    s_bio  = np.concatenate(bio_oof)
    s_bt   = np.concatenate(biotopo_oof)
    return {"per_repeat":per_repeat,"summary":summary,"delong_ready":{"y":y_pool,"bio":s_bio,"biotopo":s_bt}}


def _compute_midrank(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    z = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        # midrank for the tie block [i, j)
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out

def _fast_delong(scores, y):
    """
    scores: array (n_models, n_samples)
    y     : binary (n_samples,), values in {0,1}
    Returns: aucs (n_models,), covariance matrix (n_models, n_models)
    """
    y = np.asarray(y, dtype=int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    assert m > 0 and n > 0, "Need both classes."

    n_models, n_samples = scores.shape
    # pooled midranks per model
    tx = np.zeros((n_models, m), dtype=float)  # pooled midranks at positive indices
    ty = np.zeros((n_models, n), dtype=float)  # pooled midranks at negative indices

    for i in range(n_models):
        t = _compute_midrank(scores[i, :])      # midranks on the pooled samples
        tx[i, :] = t[pos]
        ty[i, :] = t[neg]

    # AUCs
    aucs = (tx.sum(axis=1) / m - (m + 1) / 2.0) / n

    # Variance components (Sun & Xu 2014)
    v01 = (tx - (m + 1) / 2.0) / n   # shape (n_models, m)
    v10 = (ty - (n + 1) / 2.0) / m   # shape (n_models, n)

    s01 = np.cov(v01, bias=False)    # (n_models, n_models)
    s10 = np.cov(v10, bias=False)
    S = s01 / m + s10 / n
    return aucs, S

def delong_roc_test(y_true, scores_a, scores_b, alpha=0.95):
    y_true = np.asarray(y_true, dtype=int)
    s_a = np.asarray(scores_a, dtype=float).reshape(1, -1)
    s_b = np.asarray(scores_b, dtype=float).reshape(1, -1)
    scores = np.vstack([s_a, s_b])
    if scores.shape[1] != y_true.shape[0]:
        raise ValueError("scores and y_true length mismatch")
    if not (np.any(y_true == 0) and np.any(y_true == 1)):
        raise ValueError("y_true must contain both classes")

    aucs, cov = _fast_delong(scores, y_true)
    auc_a, auc_b = aucs[0], aucs[1]
    delta = auc_b - auc_a
    var   = cov[0,0] + cov[1,1] - 2.0 * cov[0,1]
    var   = max(var, 1e-12)  # guard against tiny negatives
    se    = sqrt(var)
    z     = delta / se
    p     = 2.0 * (1.0 - norm.cdf(abs(z)))

    zcrit = norm.ppf(0.5 + alpha/2.0)
    ci_lo, ci_hi = delta - zcrit * se, delta + zcrit * se

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta": float(delta),
        "var":   float(var),
        "z":     float(z),
        "p":     float(p),
        "ci_low": float(ci_lo),
        "ci_high": float(ci_hi),
    }


import matplotlib.pyplot as plt
import numpy as np

def plot_metric_bars(summary_dict,
                     model_labels=("Bio","Bio+Topo"),
                     metrics=("AUC","AP","F1","ACC"),
                     figsize=(6,4),
                     colors=("tab:blue","tab:orange"),
                     title=None,
                     ylim=None):


    mA, mB = model_labels
    x = np.arange(len(metrics))
    width = 0.35

    # Gather means & stds
    means_A = [summary_dict[mA][f"{m}_mean"] for m in metrics]
    stds_A  = [summary_dict[mA][f"{m}_std"]  for m in metrics]
    means_B = [summary_dict[mB][f"{m}_mean"] for m in metrics]
    stds_B  = [summary_dict[mB][f"{m}_std"]  for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width/2, means_A, width, yerr=stds_A,
           label=mA, color=colors[0], capsize=4)
    ax.bar(x + width/2, means_B, width, yerr=stds_B,
           label=mB, color=colors[1], capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(ylim if ylim else (0, 1.05 * max(means_A + means_B)))
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig, ax





class LearnableAlphaPairModel(nn.Module):
    """
    Build per-drug fused vectors as [bio, alpha*topo], then pair with either concat or sym4.
    alpha is a single learnable scalar shared across drugs.
    """
    def __init__(self, d_bio, d_topo, mode="concat", hidden=(256,128), pdrop=0.1, alpha_init=1.0):
        super().__init__()
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        d_fused = d_bio + d_topo
        in_dim = (2*d_fused) if mode=="concat" else (4*d_fused)
        self.head = MLP(in_dim, hidden, pdrop)

    def fuse(self, b, t):
        return torch.cat([b, self.alpha * t], dim=1)

    def pair(self, f1, f2):
        if self.mode == "concat":
            return torch.cat([f1, f2], dim=1)
        else:
            return torch.cat([f1, f2, torch.abs(f1-f2), f1*f2], dim=1)

    def forward(self, b1, t1, b2, t2):
        f1 = self.fuse(b1, t1)
        f2 = self.fuse(b2, t2)
        x  = self.pair(f1, f2)
        return self.head(x)



def run_cv_bio_vs_learnable_alpha_topo(
    Xbio, Xtopo, y, mode="concat",
    n_splits=5, n_repeats=5, undersample=True, base_seed=13,
    head_cfg=None, alpha_lr_mult=20.0, alpha_init=1.0, device="cpu"
):
    """
    Paired CV:
      - Bio-only MLP (baseline)
      - Bio+alpha*Topo (alpha learnable scalar)
    Returns same dict shape as before (per_repeat, summary, delong_ready).
    """
    if head_cfg is None: head_cfg = {}

    per_repeat=[]
    y_all=[]; bio_oof=[]; biotopo_oof=[]

    def _avg(ms):
        keys = ms[0].keys()
        return {k: float(np.mean([d[k] for d in ms])) for k in keys}
    def _summary(per_repeat,key):
        ks = list(per_repeat[0][key].keys()); out={}
        for m in ks:
            vals = np.array([rep[key][m] for rep in per_repeat], dtype=float)
            out[f"{m}_mean"]=float(vals.mean()); out[f"{m}_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        return out

    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed+100*r)
        fold_m_bio=[]; fold_m_bt=[]
        for f,(tr,va) in enumerate(skf.split(np.zeros_like(y), y)):
            y_tr_, y_va = y[tr], y[va]
            Xb_tr_, Xb_va = Xbio[tr], Xbio[va]
            Xt_tr_, Xt_va = Xtopo[tr], Xtopo[va]

            if undersample:
                keep = undersample_majority_idx(y_tr_, seed=base_seed+r+f)
                y_tr_ = y_tr_[keep]; Xb_tr_ = Xb_tr_[keep]; Xt_tr_ = Xt_tr_[keep]

            # scale per fold with existing helper (BIO: standard, TOPO: minmax)
            (b1_tr,b2_tr,Db,b1_va,b2_va), (t1_tr,t2_tr,Dt,t1_va,t2_va) = _fit_transform_blocks(
                Xb_tr_, Xt_tr_, Xb_va, Xt_va
            )

            # ---- BIO-only baseline ----
            def build_pair(mode, a1,a2):
                if mode=="concat":
                    return np.concatenate([a1,a2],1)
                else:
                    return np.concatenate([a1,a2, np.abs(a1-a2), a1*a2],1)

            Xtr_bio = build_pair(mode, b1_tr, b2_tr)
            Xva_bio = build_pair(mode, b1_va, b2_va)
            cfgA = dict(head_cfg); cfgA["seed"]=head_cfg.get("seed",base_seed)+17*(r+1)+f
            p_bio,_ = train_mlp_auc_earlystop(Xtr_bio, y_tr_, Xva_bio, y_va, cfg=cfgA)
            fold_m_bio.append(eval_metrics(y_va, p_bio))

            # ---- Learnable-alpha model ----
            model = LearnableAlphaPairModel(
                d_bio=b1_tr.shape[1], d_topo=t1_tr.shape[1],
                mode=mode, hidden=head_cfg.get("hidden",(256,128)),
                pdrop=head_cfg.get("pdrop",0.1), alpha_init=alpha_init
            ).to(device)

            # Two param groups: alpha with higher LR & no WD
            params_alpha = [p for n,p in model.named_parameters() if n=="alpha"]
            params_main  = [p for n,p in model.named_parameters() if n!="alpha"]
            opt = torch.optim.AdamW([
                {"params": params_main,  "lr": head_cfg.get("lr",1e-3), "weight_decay": head_cfg.get("weight_decay",1e-4)},
                {"params": params_alpha, "lr": head_cfg.get("lr",1e-3)*alpha_lr_mult, "weight_decay": 0.0},
            ])
            lossf = nn.BCEWithLogitsLoss()

            ds = torch.utils.data.TensorDataset(
                torch.tensor(b1_tr, dtype=torch.float32),
                torch.tensor(t1_tr, dtype=torch.float32),
                torch.tensor(b2_tr, dtype=torch.float32),
                torch.tensor(t2_tr, dtype=torch.float32),
                torch.tensor(y_tr_, dtype=torch.float32)
            )
            dl = torch.utils.data.DataLoader(ds, batch_size=head_cfg.get("batch_size",256), shuffle=True)

            best_auc=-1; best_state=None; wait=0
            for ep in range(head_cfg.get("epochs",20)):
                model.train()
                for b1,t1,b2,t2, yb in dl:
                    b1,t1,b2,t2, yb = b1.to(device),t1.to(device),b2.to(device),t2.to(device), yb.to(device)
                    logits = model(b1,t1,b2,t2)
                    loss = lossf(logits, yb)
                    opt.zero_grad(); loss.backward(); opt.step()

                # val
                model.eval()
                with torch.no_grad():
                    b1v = torch.tensor(b1_va, dtype=torch.float32, device=device)
                    t1v = torch.tensor(t1_va, dtype=torch.float32, device=device)
                    b2v = torch.tensor(b2_va, dtype=torch.float32, device=device)
                    t2v = torch.tensor(t2_va, dtype=torch.float32, device=device)
                    pv  = torch.sigmoid(model(b1v,t1v,b2v,t2v)).cpu().numpy()
                    auc = roc_auc_score(y_va, pv)
                if auc > best_auc + head_cfg.get("min_delta",5e-4):
                    best_auc=auc
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= head_cfg.get("patience",5):
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                b1v = torch.tensor(b1_va, dtype=torch.float32, device=device)
                t1v = torch.tensor(t1_va, dtype=torch.float32, device=device)
                b2v = torch.tensor(b2_va, dtype=torch.float32, device=device)
                t2v = torch.tensor(t2_va, dtype=torch.float32, device=device)
                p_bt = torch.sigmoid(model(b1v,t1v,b2v,t2v)).cpu().numpy()

            fold_m_bt.append(eval_metrics(y_va, p_bt))

            # collect paired OOF
            y_all.append(y_va); bio_oof.append(p_bio); biotopo_oof.append(p_bt)

        per_repeat.append({
            "Bio": _avg(fold_m_bio),
            "BioTopo": _avg(fold_m_bt),
        })

    def _summ(per_repeat, key):
        ks = list(per_repeat[0][key].keys()); out={}
        for m in ks:
            vals = np.array([rep[key][m] for rep in per_repeat], dtype=float)
            out[f"{m}_mean"]=float(vals.mean()); out[f"{m}_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        return out

    summary={"Bio":_summ(per_repeat,"Bio"), "BioTopo":_summ(per_repeat,"BioTopo")}
    y_pool = np.concatenate(y_all)
    s_bio  = np.concatenate(bio_oof)
    s_btop = np.concatenate(biotopo_oof)
    return {"per_repeat":per_repeat,"summary":summary,"delong_ready":{"y":y_pool,"bio":s_bio,"biotopo":s_btop}}
    


class Tower(nn.Module):
    def __init__(self, d_in, d_latent, hidden=(256,), pdrop=0.1):
        super().__init__()
        layers=[]; x=d_in
        for h in hidden:
            layers += [nn.Linear(x,h), nn.ReLU(), nn.Dropout(pdrop)]
            x=h
        layers += [nn.Linear(x, d_latent)]
        self.net = nn.Sequential(*layers)
        self.ln  = nn.LayerNorm(d_latent)
    def forward(self, x):
        return self.ln(self.net(x))

class LatentGatePairModel(nn.Module):
    """
    Two-tower encoders (bio/topo) -> gated fusion per drug -> pair (concat|sym4) -> head MLP.
    Gate g \in (0,1)^d is learned from [z_b || z_t].
    """
    def __init__(self, d_bio, d_topo, d_latent=128, mode="concat",
                 enc_hidden=(256,), head_hidden=(256,128), pdrop=0.1,
                 gate_scalar=False, gate_bias_init=0.7):
        super().__init__()
        self.mode = mode
        self.bio = Tower(d_bio, d_latent, hidden=enc_hidden, pdrop=pdrop)
        self.topo= Tower(d_topo, d_latent, hidden=enc_hidden, pdrop=pdrop)

        g_out = 1 if gate_scalar else d_latent
        self.gate = nn.Linear(2*d_latent, g_out)
        # initialize gate bias to favor topo (sigmoid^-1(0.7) ≈ 0.847)
        with torch.no_grad():
            self.gate.bias.fill_(torch.logit(torch.tensor(gate_bias_init)))

        in_dim = (2*d_latent) if mode=="concat" else (4*d_latent)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden[0]), nn.ReLU(), nn.Dropout(pdrop),
            *sum(([nn.Linear(head_hidden[i], head_hidden[i+1]), nn.ReLU(), nn.Dropout(pdrop)]
                  for i in range(len(head_hidden)-1)), []),
            nn.Linear(head_hidden[-1], 1)
        )

    def fuse_one(self, b, t):
        z_b = self.bio(b)             # [B, d]
        z_t = self.topo(t)            # [B, d]
        g   = torch.sigmoid(self.gate(torch.cat([z_b, z_t], dim=1)))   # [B, d] or [B,1]
        if g.shape[1] == 1: g = g.expand_as(z_b)
        z = (1 - g) * z_b + g * z_t
        return z, g

    def pair(self, f1, f2):
        if self.mode == "concat":
            return torch.cat([f1, f2], dim=1)
        else:
            return torch.cat([f1, f2, torch.abs(f1-f2), f1*f2], dim=1)

    def forward(self, b1, t1, b2, t2):
        f1, g1 = self.fuse_one(b1, t1)
        f2, g2 = self.fuse_one(b2, t2)
        x = self.pair(f1, f2)
        logit = self.head(x).squeeze(-1)
        return logit, (g1, g2)    