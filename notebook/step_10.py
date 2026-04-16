# %%
# Step 10: VAE Imputation Ablation Study
# Full 3x2 ablation: (Raw / MLP-VAE / GRU-VAE) x (no / with missingness indicators)
# Decomposes contributions of VAE architecture and missingness indicators.

import matplotlib
matplotlib.use("Agg")

import os
import json
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    accuracy_score, f1_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay,
)

RANDOM_SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "temporal")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
# ============================================================
# Part A: Load data
# ============================================================

print("=== Part A: Load data ===\n")

df = pd.read_csv("final_dataset.csv")
print(f"Dataset: {df.shape}")

y_48 = df["label_Y48"].values.astype(np.float32)
y_72 = df["label_Y72"].values.astype(np.float32)
print(f"Y48 events: {int(y_48.sum())}/{len(y_48)}")
print(f"Y72 events: {int(y_72.sum())}/{len(y_72)}")

# %%
# === Build windowed MEAN VALUES from raw time series ===

TEMPORAL_VARS = [
    "hr", "rr", "ventilation_flag", "urine_output", "sofa_cardio",
    "gcs", "temp", "sofa_cns", "sofa_renal", "sodium", "creatinine", "bun",
    "platelets", "vasopressor_dose", "sofa_coag", "map", "lactate", "sofa_resp",
    "bilirubin", "sofa_liver", "fluid_input", "wbc",
]

print("Loading time series for windowed mean values...")
ts = pd.read_csv("timeseries_relative.csv")
ts = ts[ts["stay_id"].isin(set(df["stay_id"]))]
ts = ts[(ts["relative_hour"] >= 0) & (ts["relative_hour"] <= 24)]

ts["window"] = pd.cut(
    ts["relative_hour"], bins=[0, 6, 12, 18, 24],
    labels=[0, 1, 2, 3], right=True, include_lowest=True,
).astype(int)

available_vars = [v for v in TEMPORAL_VARS if v in ts.columns]
print(f"Temporal variables: {len(available_vars)}")

# %%
# Compute mean value per (stay_id, window, variable) — NaN where no measurements
stay_ids = df["stay_id"].values
sid_index = pd.Index(stay_ids, name="stay_id")

windowed_means = pd.DataFrame(index=sid_index)
for w in range(4):
    ts_w = ts[ts["window"] == w]
    for var in available_vars:
        means = ts_w.dropna(subset=[var]).groupby("stay_id")[var].mean()
        windowed_means[f"{var}_mean_w{w}"] = means.reindex(sid_index)
print(f"Windowed mean features: {windowed_means.shape[1]} columns")

# %%
# Build temporal tensor (N, 4 windows, F variables) with NaN for missing
N = len(df)
T = 4
F = len(available_vars)
X_t = np.full((N, T, F), np.nan, dtype=np.float32)

for w in range(T):
    for fi, var in enumerate(available_vars):
        X_t[:, w, fi] = windowed_means[f"{var}_mean_w{w}"].values

# Per-variable missingness mask: (N, T, F) — True where variable has no measurements
M_t = np.isnan(X_t)

# Per-window mask for BiGRU attention: True if ALL vars missing in that window
M_window = M_t.all(axis=2)  # (N, T)

print(f"Temporal tensor: {X_t.shape}")
print(f"Missing variable-cells: {M_t.sum()} / {M_t.size} ({100 * M_t.mean():.1f}%)")
print(f"Fully missing windows:  {M_window.sum()} / {M_window.size} ({100 * M_window.mean():.1f}%)")

# %%
# train/val/test split
NON_FEATURE_COLS = ["subject_id", "stay_id", "label_Y48", "label_Y72", "split"]
train_mask = df["split"] == "train"
test_mask = df["split"] == "test"
all_train_idx = np.where(train_mask.values)[0]
test_idx = np.where(test_mask.values)[0]

train_idx, val_idx = train_test_split(
    all_train_idx, test_size=0.15, random_state=RANDOM_SEED,
    stratify=y_48[all_train_idx],
)
print(f"Split -> train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")

# %%
# Normalize temporal features (fit on train observed values only, then zero-fill NaN)
X_train_flat = X_t[train_idx].reshape(-1, F)
feat_mean = np.nanmean(X_train_flat, axis=0)
feat_std = np.nanstd(X_train_flat, axis=0)
feat_std[feat_std == 0] = 1.0


def scale_and_fill(X, mean, std):
    """Standardize and fill NaN with 0 (= population mean after scaling)."""
    X_scaled = (X - mean) / std
    return np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)


Xt_tr = scale_and_fill(X_t[train_idx].reshape(-1, F), feat_mean, feat_std).reshape(-1, T, F)
Xt_va = scale_and_fill(X_t[val_idx].reshape(-1, F), feat_mean, feat_std).reshape(-1, T, F)
Xt_te = scale_and_fill(X_t[test_idx].reshape(-1, F), feat_mean, feat_std).reshape(-1, T, F)

T_DIM = F
print(f"T_DIM={T_DIM}")

# %%
# ============================================================
# Part B: VAE Architectures
# ============================================================

print("\n=== Part B: VAE Architectures ===\n")


class MLPVAE(nn.Module):
    """Per-timestep MLP-based VAE. Processes each window independently."""
    def __init__(self, feat_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class SequentialVAE(nn.Module):
    """Sequence-aware GRU-based VAE. Processes all windows together
    so cross-window temporal patterns inform imputation."""
    def __init__(self, feat_dim, n_windows=4, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.n_windows = n_windows
        self.enc_gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)
        self.dec_proj = nn.Linear(latent_dim, hidden_dim)
        self.dec_gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1,
            batch_first=True,
        )
        self.dec_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def encode(self, x):
        _, h_n = self.enc_gru(x)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        z_proj = self.dec_proj(z)
        z_rep = z_proj.unsqueeze(1).expand(-1, self.n_windows, -1)
        dec_out, _ = self.dec_gru(z_rep)
        return self.dec_out(dec_out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss_fn(x_recon, x_orig, mu, logvar, obs_mask, beta=0.5):
    """VAE loss computed only on observed (non-missing) cells."""
    obs_float = obs_mask.float()
    n_obs = obs_float.sum()
    if n_obs > 0:
        recon = ((x_recon - x_orig) ** 2 * obs_float).sum() / n_obs
    else:
        recon = torch.tensor(0.0, device=x_recon.device)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


# %%
# ============================================================
# Part C: Train both VAEs
# ============================================================

print("=== Part C: Train VAEs ===\n")


def train_mlp_vae(Xt_tr, M_t_train, feat_dim, max_epochs=150, patience=15):
    """Train per-timestep MLP-VAE."""
    vae = MLPVAE(feat_dim=feat_dim, latent_dim=16, hidden_dim=64).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    X_flat = torch.FloatTensor(Xt_tr).view(-1, feat_dim)
    obs_flat = torch.BoolTensor(~M_t_train).reshape(-1, feat_dim)
    loader = DataLoader(TensorDataset(X_flat, obs_flat), batch_size=64, shuffle=True)

    best_loss, best_state, no_imp = float("inf"), None, 0
    for epoch in range(max_epochs):
        vae.train()
        ep_loss = 0
        for batch_x, batch_obs in loader:
            batch_x, batch_obs = batch_x.to(device), batch_obs.to(device)
            opt.zero_grad()
            x_rec, mu, lv = vae(batch_x)
            loss, _, _ = vae_loss_fn(x_rec, batch_x, mu, lv, batch_obs)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        if avg < best_loss:
            best_loss, best_state, no_imp = avg, copy.deepcopy(vae.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"  MLP-VAE: early stop at epoch {epoch+1}, loss={best_loss:.4f}")
            break
    else:
        print(f"  MLP-VAE: finished {max_epochs} epochs, loss={best_loss:.4f}")
    vae.load_state_dict(best_state)
    return vae


def train_gru_vae(Xt_tr, M_t_train, feat_dim, n_windows, max_epochs=200, patience=20):
    """Train sequence-aware GRU-VAE."""
    vae = SequentialVAE(feat_dim=feat_dim, n_windows=n_windows, latent_dim=32, hidden_dim=64).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    X_seq = torch.FloatTensor(Xt_tr)
    obs_seq = torch.BoolTensor(~M_t_train)
    loader = DataLoader(TensorDataset(X_seq, obs_seq), batch_size=256, shuffle=True)

    best_loss, best_state, no_imp = float("inf"), None, 0
    for epoch in range(max_epochs):
        vae.train()
        ep_loss = 0
        for batch_x, batch_obs in loader:
            batch_x, batch_obs = batch_x.to(device), batch_obs.to(device)
            opt.zero_grad()
            x_rec, mu, lv = vae(batch_x)
            loss, _, _ = vae_loss_fn(x_rec, batch_x, mu, lv, batch_obs)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        if avg < best_loss:
            best_loss, best_state, no_imp = avg, copy.deepcopy(vae.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"  GRU-VAE: early stop at epoch {epoch+1}, loss={best_loss:.4f}")
            break
    else:
        print(f"  GRU-VAE: finished {max_epochs} epochs, loss={best_loss:.4f}")
    vae.load_state_dict(best_state)
    return vae


print("Training MLP-VAE (per-timestep)...")
t0 = time.time()
mlp_vae = train_mlp_vae(Xt_tr, M_t[train_idx], F)
print(f"  Time: {time.time() - t0:.1f}s\n")

print("Training GRU-VAE (sequence-aware)...")
t0 = time.time()
gru_vae = train_gru_vae(Xt_tr, M_t[train_idx], F, T)
print(f"  Time: {time.time() - t0:.1f}s")

# %%
# ============================================================
# Part D: Apply imputations from both VAEs
# ============================================================

print("\n=== Part D: Apply imputations ===\n")


def apply_mlp_imputation(X_np, mask_np, vae):
    """MLP-VAE imputation: process each timestep independently."""
    N, T, F = X_np.shape
    X_flat = torch.FloatTensor(X_np).to(device).view(N * T, F)
    mask_flat = torch.BoolTensor(mask_np).to(device).view(N * T, F)
    vae.eval()
    with torch.no_grad():
        x_recon, _, _ = vae(X_flat)
        X_imp = torch.where(mask_flat, x_recon, X_flat)
    return X_imp.view(N, T, F).cpu().numpy()


def apply_gru_imputation(X_np, mask_np, vae, batch_size=512):
    """GRU-VAE imputation: process full patient sequences."""
    N = X_np.shape[0]
    X_imp = X_np.copy()
    vae.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = torch.FloatTensor(X_np[start:end]).to(device)
            M_batch = torch.BoolTensor(mask_np[start:end]).to(device)
            x_recon, _, _ = vae(X_batch)
            X_imp[start:end] = torch.where(M_batch, x_recon, X_batch).cpu().numpy()
    return X_imp


# MLP-VAE imputed data
Xt_tr_mlp = apply_mlp_imputation(Xt_tr, M_t[train_idx], mlp_vae)
Xt_va_mlp = apply_mlp_imputation(Xt_va, M_t[val_idx], mlp_vae)
Xt_te_mlp = apply_mlp_imputation(Xt_te, M_t[test_idx], mlp_vae)

# GRU-VAE imputed data
Xt_tr_gru = apply_gru_imputation(Xt_tr, M_t[train_idx], gru_vae)
Xt_va_gru = apply_gru_imputation(Xt_va, M_t[val_idx], gru_vae)
Xt_te_gru = apply_gru_imputation(Xt_te, M_t[test_idx], gru_vae)

# Verify both imputations
for label, Xt_imp, mask in [("MLP-VAE", Xt_tr_mlp, M_t[train_idx]),
                            ("GRU-VAE", Xt_tr_gru, M_t[train_idx])]:
    diff = np.abs(Xt_imp - Xt_tr)
    print(f"  {label}: imputed cell change={diff[mask].mean():.4f}, "
          f"observed cell change={diff[~mask].mean():.6f}")

# %%
# ============================================================
# Part E: Prepare all 6 input conditions
# ============================================================

print("\n=== Part E: Prepare ablation inputs ===\n")

miss_tr = M_t[train_idx].astype(np.float32)
miss_va = M_t[val_idx].astype(np.float32)
miss_te = M_t[test_idx].astype(np.float32)

# 6 conditions: (imputation_method) x (miss_indicators)
conditions = {}

# 1. Raw, no miss indicators
conditions["Raw"] = {
    "tr": Xt_tr, "va": Xt_va, "te": Xt_te, "input_dim": F,
}

# 2. Raw, with miss indicators
conditions["Raw+Miss"] = {
    "tr": np.concatenate([Xt_tr, miss_tr], axis=2),
    "va": np.concatenate([Xt_va, miss_va], axis=2),
    "te": np.concatenate([Xt_te, miss_te], axis=2),
    "input_dim": 2 * F,
}

# 3. MLP-VAE, no miss indicators
conditions["MLP-VAE"] = {
    "tr": Xt_tr_mlp, "va": Xt_va_mlp, "te": Xt_te_mlp, "input_dim": F,
}

# 4. MLP-VAE, with miss indicators
conditions["MLP-VAE+Miss"] = {
    "tr": np.concatenate([Xt_tr_mlp, miss_tr], axis=2),
    "va": np.concatenate([Xt_va_mlp, miss_va], axis=2),
    "te": np.concatenate([Xt_te_mlp, miss_te], axis=2),
    "input_dim": 2 * F,
}

# 5. GRU-VAE, no miss indicators
conditions["GRU-VAE"] = {
    "tr": Xt_tr_gru, "va": Xt_va_gru, "te": Xt_te_gru, "input_dim": F,
}

# 6. GRU-VAE, with miss indicators
conditions["GRU-VAE+Miss"] = {
    "tr": np.concatenate([Xt_tr_gru, miss_tr], axis=2),
    "va": np.concatenate([Xt_va_gru, miss_va], axis=2),
    "te": np.concatenate([Xt_te_gru, miss_te], axis=2),
    "input_dim": 2 * F,
}

for name, cond in conditions.items():
    print(f"  {name:<20} input_dim={cond['input_dim']}")

# %%
# ============================================================
# Part F: BiGRU Classifier
# ============================================================


class TemporalDataset(Dataset):
    def __init__(self, X_t, mask, y48, y72):
        self.X_t = torch.FloatTensor(X_t)
        self.mask = torch.BoolTensor(mask)
        self.y48 = torch.FloatTensor(y48)
        self.y72 = torch.FloatTensor(y72)
    def __len__(self): return len(self.y48)
    def __getitem__(self, i): return self.X_t[i], self.mask[i], self.y48[i], self.y72[i]


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, x, mask=None):
        s = self.w(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(mask, -1e9)
        w = torch.softmax(s, dim=-1)
        return (w.unsqueeze(-1) * x).sum(1)


class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.pool = AttnPool(d_model)
        self.h48 = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.h72 = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        x = self.norm(self.proj(x))
        out, _ = self.gru(x)
        pooled = self.pool(out, mask)
        return self.h48(pooled).squeeze(-1), self.h72(pooled).squeeze(-1)


# %%
# ============================================================
# Part G: Train + Evaluate all conditions
# ============================================================

def best_f1_threshold(y_true, y_prob):
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1 = 2 * pr * rc / (pr + rc + 1e-8)
    return th[np.argmax(f1[:-1])]


def train_eval_bigru(cond, label, epochs=50, patience=10):
    tr_dl = DataLoader(
        TemporalDataset(cond["tr"], M_window[train_idx], y_48[train_idx], y_72[train_idx]),
        batch_size=32, shuffle=True)
    va_dl = DataLoader(
        TemporalDataset(cond["va"], M_window[val_idx], y_48[val_idx], y_72[val_idx]),
        batch_size=64, shuffle=False)
    te_dl = DataLoader(
        TemporalDataset(cond["te"], M_window[test_idx], y_48[test_idx], y_72[test_idx]),
        batch_size=64, shuffle=False)

    model = BiGRUClassifier(cond["input_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    pw48 = torch.tensor([(y_48 == 0).sum() / (y_48 == 1).sum()]).to(device)
    pw72 = torch.tensor([(y_72 == 0).sum() / (y_72 == 1).sum()]).to(device)
    c48 = nn.BCEWithLogitsLoss(pos_weight=pw48)
    c72 = nn.BCEWithLogitsLoss(pos_weight=pw72)

    best_auc, no_imp, best_state = 0, 0, None

    for ep in range(epochs):
        model.train()
        for x_t, mask, y48, y72 in tr_dl:
            x_t, mask = x_t.to(device), mask.to(device)
            y48, y72 = y48.to(device), y72.to(device)
            opt.zero_grad()
            l48, l72 = model(x_t, mask)
            loss = c48(l48, y48) + c72(l72, y72)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        p48v, p72v, t48v, t72v = [], [], [], []
        with torch.no_grad():
            for x_t, mask, y48, y72 in va_dl:
                l48, l72 = model(x_t.to(device), mask.to(device))
                p48v += torch.sigmoid(l48).cpu().tolist()
                p72v += torch.sigmoid(l72).cpu().tolist()
                t48v += y48.tolist()
                t72v += y72.tolist()
        mean_auc = (roc_auc_score(t48v, p48v) + roc_auc_score(t72v, p72v)) / 2
        if mean_auc > best_auc:
            best_auc, best_state, no_imp = mean_auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    p48, p72, t48, t72 = [], [], [], []
    with torch.no_grad():
        for x_t, mask, y48, y72 in te_dl:
            l48, l72 = model(x_t.to(device), mask.to(device))
            p48 += torch.sigmoid(l48).cpu().tolist()
            p72 += torch.sigmoid(l72).cpu().tolist()
            t48 += y48.tolist()
            t72 += y72.tolist()

    p48, p72 = np.array(p48), np.array(p72)
    t48, t72 = np.array(t48), np.array(t72)

    th48 = best_f1_threshold(t48, p48)
    th72 = best_f1_threshold(t72, p72)
    pred48 = (p48 >= th48).astype(int)
    pred72 = (p72 >= th72).astype(int)

    auroc48 = roc_auc_score(t48, p48)
    auroc72 = roc_auc_score(t72, p72)
    auprc48 = average_precision_score(t48, p48)
    auprc72 = average_precision_score(t72, p72)
    acc48 = accuracy_score(t48, pred48)
    acc72 = accuracy_score(t72, pred72)
    f1_48 = f1_score(t48, pred48, zero_division=0)
    f1_72 = f1_score(t72, pred72, zero_division=0)

    print(f"  {label:<20} Y48 AUROC:{auroc48:.4f}  Y72 AUROC:{auroc72:.4f}  "
          f"mean:{(auroc48+auroc72)/2:.4f}")

    return {
        "label": label, "model": model,
        "auroc48": auroc48, "auroc72": auroc72,
        "auprc48": auprc48, "auprc72": auprc72,
        "acc48": acc48, "acc72": acc72,
        "f1_48": f1_48, "f1_72": f1_72,
        "p48": p48, "p72": p72,
        "t48": t48, "t72": t72,
        "pred48": pred48, "pred72": pred72,
        "th48": th48, "th72": th72,
    }


# %%
print("\n=== Part G: Train BiGRU for all 6 conditions ===\n")

all_results = {}
for name, cond in conditions.items():
    t0 = time.time()
    all_results[name] = train_eval_bigru(cond, name)
    elapsed = time.time() - t0
    print(f"  {'':20} Time: {elapsed:.1f}s\n")

# %%
# ============================================================
# Part H: Save results
# ============================================================

print("=== Part H: Save results ===\n")


def save_results(res, name):
    for suffix, y_key, p_key in [("label_Y48", "t48", "p48"), ("label_Y72", "t72", "p72")]:
        y_true = res[y_key]
        y_prob = res[p_key]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        metrics = {"model": name, "label": suffix, "test_auc": float(auc), "test_auprc": float(ap)}
        with open(os.path.join(RESULTS_DIR, f"{name}_{suffix}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        np.savez(os.path.join(RESULTS_DIR, f"{name}_{suffix}_predictions.npz"),
                 y_true=y_true, y_prob=y_prob, y_pred=y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name=name)
        axes[0].set_title(f"ROC Curve ({suffix})")
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name=name)
        axes[1].set_title(f"PR Curve ({suffix})")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}_{suffix}_curves.png"), dpi=150)
        plt.close()


for name, res in all_results.items():
    save_results(res, name)
    print(f"  Saved {name}")

# Save VAE models
torch.save(mlp_vae.state_dict(), os.path.join(RESULTS_DIR, "mlp_vae_model.pt"))
torch.save(gru_vae.state_dict(), os.path.join(RESULTS_DIR, "gru_vae_model.pt"))
print("  Saved VAE model weights")

# %%
# ============================================================
# Part I: Ablation Comparison
# ============================================================

print("\n=== Part I: Ablation Results ===\n")

# Full results table
print(f"  {'Condition':<20} {'AUROC48':>9} {'AUROC72':>9} {'Mean':>9} "
      f"{'AUPRC48':>9} {'AUPRC72':>9} {'F1_48':>8} {'F1_72':>8}")
print(f"  {'-'*85}")
for name, res in all_results.items():
    mean_auc = (res["auroc48"] + res["auroc72"]) / 2
    print(f"  {name:<20} {res['auroc48']:>9.4f} {res['auroc72']:>9.4f} {mean_auc:>9.4f} "
          f"{res['auprc48']:>9.4f} {res['auprc72']:>9.4f} {res['f1_48']:>8.4f} {res['f1_72']:>8.4f}")

# Decomposition
print(f"\n{'='*70}")
print(f"  EFFECT DECOMPOSITION (Mean AUROC)")
print(f"{'='*70}")

def mean_auroc(name):
    r = all_results[name]
    return (r["auroc48"] + r["auroc72"]) / 2

baseline = mean_auroc("Raw")
print(f"\n  Baseline (Raw, no miss ind):          {baseline:.4f}")

# Effect of missingness indicators alone (on raw data)
miss_effect = mean_auroc("Raw+Miss") - mean_auroc("Raw")
print(f"\n  --- Effect of missingness indicators ---")
print(f"  Raw        -> Raw+Miss:               {miss_effect:+.4f}")
print(f"  MLP-VAE    -> MLP-VAE+Miss:           {mean_auroc('MLP-VAE+Miss') - mean_auroc('MLP-VAE'):+.4f}")
print(f"  GRU-VAE    -> GRU-VAE+Miss:           {mean_auroc('GRU-VAE+Miss') - mean_auroc('GRU-VAE'):+.4f}")

# Effect of VAE imputation alone (without miss indicators)
print(f"\n  --- Effect of VAE imputation (no miss ind) ---")
print(f"  Raw        -> MLP-VAE:                {mean_auroc('MLP-VAE') - mean_auroc('Raw'):+.4f}")
print(f"  Raw        -> GRU-VAE:                {mean_auroc('GRU-VAE') - mean_auroc('Raw'):+.4f}")

# Effect of VAE imputation (with miss indicators)
print(f"\n  --- Effect of VAE imputation (with miss ind) ---")
print(f"  Raw+Miss   -> MLP-VAE+Miss:           {mean_auroc('MLP-VAE+Miss') - mean_auroc('Raw+Miss'):+.4f}")
print(f"  Raw+Miss   -> GRU-VAE+Miss:           {mean_auroc('GRU-VAE+Miss') - mean_auroc('Raw+Miss'):+.4f}")

# Effect of GRU vs MLP architecture
print(f"\n  --- Effect of GRU vs MLP architecture ---")
print(f"  MLP-VAE    -> GRU-VAE:                {mean_auroc('GRU-VAE') - mean_auroc('MLP-VAE'):+.4f}")
print(f"  MLP-VAE+Miss -> GRU-VAE+Miss:         {mean_auroc('GRU-VAE+Miss') - mean_auroc('MLP-VAE+Miss'):+.4f}")

# Best overall
best_name = max(all_results.keys(), key=mean_auroc)
print(f"\n  Best condition: {best_name} ({mean_auroc(best_name):.4f})")
print(f"  Improvement over baseline: {mean_auroc(best_name) - baseline:+.4f}")
print(f"{'='*70}")

# %%
# Save ablation summary CSV
rows = []
for name, res in all_results.items():
    rows.append({
        "condition": name,
        "auroc_Y48": res["auroc48"], "auroc_Y72": res["auroc72"],
        "mean_auroc": (res["auroc48"] + res["auroc72"]) / 2,
        "auprc_Y48": res["auprc48"], "auprc_Y72": res["auprc72"],
        "f1_Y48": res["f1_48"], "f1_Y72": res["f1_72"],
    })
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
print(f"\nSaved ablation_results.csv to {RESULTS_DIR}")

# %%
print("\n=== Step 10 complete ===")
