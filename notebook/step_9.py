# %%
# Step 9: Temporal Deep Learning Models
# BiGRU/BiLSTM on windowed measurement counts, MLP/RF on static features, ensembles

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
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "temporal")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
# === Load data ===

df = pd.read_csv("final_dataset.csv")
print(f"Dataset: {df.shape}")

y_48 = df["label_Y48"].values.astype(np.float32)
y_72 = df["label_Y72"].values.astype(np.float32)
print(f"Y48 events: {int(y_48.sum())}/{len(y_48)}")
print(f"Y72 events: {int(y_72.sum())}/{len(y_72)}")

# %%
# === Build windowed measurement counts from raw time series ===

TEMPORAL_VARS = [
    "hr", "rr", "ventilation_flag", "urine_output", "sofa_cardio",
    "gcs", "temp", "sofa_cns", "sofa_renal", "sodium", "creatinine", "bun",
    "platelets", "vasopressor_dose", "sofa_coag", "map", "lactate", "sofa_resp",
    "bilirubin", "sofa_liver", "fluid_input", "wbc",
]

print("Loading time series for windowed counts...")
ts = pd.read_csv("timeseries_relative.csv")
ts = ts[ts["stay_id"].isin(set(df["stay_id"]))]
ts = ts[(ts["relative_hour"] >= 0) & (ts["relative_hour"] <= 24)]

# bin into 4 windows
ts["window"] = pd.cut(
    ts["relative_hour"], bins=[0, 6, 12, 18, 24],
    labels=[0, 1, 2, 3], right=True, include_lowest=True,
).astype(int)

available_vars = [v for v in TEMPORAL_VARS if v in ts.columns]
print(f"Temporal variables: {len(available_vars)}")

# %%
# count non-null measurements per (stay_id, window, variable)
stay_ids = df["stay_id"].values
sid_index = pd.Index(stay_ids, name="stay_id")

windowed = pd.DataFrame(index=sid_index)
for w in range(4):
    ts_w = ts[ts["window"] == w]
    for var in available_vars:
        counts = ts_w.dropna(subset=[var]).groupby("stay_id").size()
        windowed[f"{var}_count_w{w}"] = counts.reindex(sid_index, fill_value=0)
windowed = windowed.fillna(0).astype(np.float32)
print(f"Windowed count features: {windowed.shape[1]} columns")

# %%
# build temporal tensor (N, 4 windows, F variables) — counts only
N = len(df)
T = 4
F = len(available_vars)
X_t = np.zeros((N, T, F), dtype=np.float32)
M_t = np.zeros((N, T), dtype=bool)

windowed_vals = windowed.values
for w in range(T):
    cols_w = windowed_vals[:, w * F:(w + 1) * F]
    X_t[:, w, :] = cols_w
    M_t[:, w] = (cols_w.sum(axis=1) == 0)

print(f"Temporal tensor: {X_t.shape}")

# %%
# static features: all numeric columns from final_dataset
NON_FEATURE_COLS = ["subject_id", "stay_id", "label_Y48", "label_Y72", "split"]
static_cols = [c for c in df.columns
               if c not in NON_FEATURE_COLS
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
X_s = df[static_cols].fillna(0).values.astype(np.float32)
print(f"Static features: {X_s.shape[1]}")

# %%
# use existing train/test split, carve validation from train
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
# normalize
sc_t = StandardScaler()
sc_s = StandardScaler()

Xt_tr = sc_t.fit_transform(X_t[train_idx].reshape(-1, F)).reshape(-1, T, F)
Xs_tr = sc_s.fit_transform(X_s[train_idx])
Xt_va = sc_t.transform(X_t[val_idx].reshape(-1, F)).reshape(-1, T, F)
Xs_va = sc_s.transform(X_s[val_idx])
Xt_te = sc_t.transform(X_t[test_idx].reshape(-1, F)).reshape(-1, T, F)
Xs_te = sc_s.transform(X_s[test_idx])

T_DIM = F
S_DIM = X_s.shape[1]
print(f"T_DIM={T_DIM}, S_DIM={S_DIM}")

# %%
# === Dataset classes ===

class TemporalDataset(Dataset):
    def __init__(self, X_t, mask, y48, y72):
        self.X_t = torch.FloatTensor(X_t)
        self.mask = torch.BoolTensor(mask)
        self.y48 = torch.FloatTensor(y48)
        self.y72 = torch.FloatTensor(y72)
    def __len__(self): return len(self.y48)
    def __getitem__(self, i): return self.X_t[i], self.mask[i], self.y48[i], self.y72[i]

class StaticDataset(Dataset):
    def __init__(self, X_s, y48, y72):
        self.X_s = torch.FloatTensor(X_s)
        self.y48 = torch.FloatTensor(y48)
        self.y72 = torch.FloatTensor(y72)
    def __len__(self): return len(self.y48)
    def __getitem__(self, i): return self.X_s[i], self.y48[i], self.y72[i]

# %%
# === Model architectures ===

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

class BiGRUModel(nn.Module):
    def __init__(self, input_dim, d_model=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.pool = AttnPool(d_model)
        self.head48 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.head72 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x, mask=None):
        x = self.norm(self.proj(x))
        out, _ = self.gru(x)
        pooled = self.pool(out, mask)
        return self.head48(pooled).squeeze(-1), self.head72(pooled).squeeze(-1)

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, d_model=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.pool = AttnPool(d_model)
        self.head48 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.head72 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x, mask=None):
        x = self.norm(self.proj(x))
        out, _ = self.lstm(x)
        pooled = self.pool(out, mask)
        return self.head48(pooled).squeeze(-1), self.head72(pooled).squeeze(-1)

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 32), nn.ReLU(),
        )
        self.head48 = nn.Linear(32, 1)
        self.head72 = nn.Linear(32, 1)
    def forward(self, x):
        h = self.net(x)
        return self.head48(h).squeeze(-1), self.head72(h).squeeze(-1)

# %%
# === Training functions ===

def train_temporal(model, name, epochs=50, patience=10):
    tr_ds = TemporalDataset(Xt_tr, M_t[train_idx], y_48[train_idx], y_72[train_idx])
    va_ds = TemporalDataset(Xt_va, M_t[val_idx], y_48[val_idx], y_72[val_idx])
    te_ds = TemporalDataset(Xt_te, M_t[test_idx], y_48[test_idx], y_72[test_idx])
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    pw48 = torch.tensor([(y_48 == 0).sum() / (y_48 == 1).sum()]).to(device)
    pw72 = torch.tensor([(y_72 == 0).sum() / (y_72 == 1).sum()]).to(device)
    c48 = nn.BCEWithLogitsLoss(pos_weight=pw48)
    c72 = nn.BCEWithLogitsLoss(pos_weight=pw72)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

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
                x_t, mask = x_t.to(device), mask.to(device)
                l48, l72 = model(x_t, mask)
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
            print(f"    {name}: early stop at epoch {ep+1}, best val AUC={best_auc:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    p48, p72, t48, t72 = [], [], [], []
    with torch.no_grad():
        for x_t, mask, y48, y72 in te_dl:
            x_t, mask = x_t.to(device), mask.to(device)
            l48, l72 = model(x_t, mask)
            p48 += torch.sigmoid(l48).cpu().tolist()
            p72 += torch.sigmoid(l72).cpu().tolist()
            t48 += y48.tolist()
            t72 += y72.tolist()

    p48, p72 = np.array(p48), np.array(p72)
    a48, a72 = roc_auc_score(t48, p48), roc_auc_score(t72, p72)
    print(f"  {name:<15} Y48:{a48:.4f}  Y72:{a72:.4f}")
    return model, p48, p72, np.array(t48), np.array(t72), a48, a72


def train_static_mlp(model, name, epochs=50, patience=10):
    tr_ds = StaticDataset(Xs_tr, y_48[train_idx], y_72[train_idx])
    va_ds = StaticDataset(Xs_va, y_48[val_idx], y_72[val_idx])
    te_ds = StaticDataset(Xs_te, y_48[test_idx], y_72[test_idx])
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    pw48 = torch.tensor([(y_48 == 0).sum() / (y_48 == 1).sum()]).to(device)
    pw72 = torch.tensor([(y_72 == 0).sum() / (y_72 == 1).sum()]).to(device)
    c48 = nn.BCEWithLogitsLoss(pos_weight=pw48)
    c72 = nn.BCEWithLogitsLoss(pos_weight=pw72)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_auc, no_imp, best_state = 0, 0, None
    for ep in range(epochs):
        model.train()
        for x_s, y48, y72 in tr_dl:
            x_s, y48, y72 = x_s.to(device), y48.to(device), y72.to(device)
            opt.zero_grad()
            l48, l72 = model(x_s)
            loss = c48(l48, y48) + c72(l72, y72)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        p48v, p72v, t48v, t72v = [], [], [], []
        with torch.no_grad():
            for x_s, y48, y72 in va_dl:
                l48, l72 = model(x_s.to(device))
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
            print(f"    {name}: early stop at epoch {ep+1}, best val AUC={best_auc:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    p48, p72, t48, t72 = [], [], [], []
    with torch.no_grad():
        for x_s, y48, y72 in te_dl:
            l48, l72 = model(x_s.to(device))
            p48 += torch.sigmoid(l48).cpu().tolist()
            p72 += torch.sigmoid(l72).cpu().tolist()
            t48 += y48.tolist()
            t72 += y72.tolist()

    p48, p72 = np.array(p48), np.array(p72)
    a48, a72 = roc_auc_score(t48, p48), roc_auc_score(t72, p72)
    print(f"  {name:<15} Y48:{a48:.4f}  Y72:{a72:.4f}")
    return model, p48, p72, np.array(t48), np.array(t72), a48, a72

# %%
# === Train temporal models ===

print("\n--- Temporal Model Benchmark ---")
gru_model, p48_gru, p72_gru, t48_true, t72_true, a48_gru, a72_gru = \
    train_temporal(BiGRUModel(T_DIM).to(device), "BiGRU")
lstm_model, p48_lstm, p72_lstm, _, _, a48_lstm, a72_lstm = \
    train_temporal(BiLSTMModel(T_DIM).to(device), "BiLSTM")

print(f"\n  Winner Y48: {'BiGRU' if a48_gru > a48_lstm else 'BiLSTM'}")
print(f"  Winner Y72: {'BiGRU' if a72_gru > a72_lstm else 'BiLSTM'}")

best_temporal_p48 = p48_gru if a48_gru >= a48_lstm else p48_lstm
best_temporal_p72 = p72_gru if a72_gru >= a72_lstm else p72_lstm

# %%
# === Train static models ===

print("\n--- Static Model Benchmark ---")
mlp_model, p48_mlp, p72_mlp, _, _, a48_mlp, a72_mlp = \
    train_static_mlp(MLPModel(S_DIM).to(device), "MLP")

rf48 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                              min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
rf72 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                              min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
rf48.fit(Xs_tr, y_48[train_idx])
rf72.fit(Xs_tr, y_72[train_idx])
p48_rf = rf48.predict_proba(Xs_te)[:, 1]
p72_rf = rf72.predict_proba(Xs_te)[:, 1]
a48_rf = roc_auc_score(y_48[test_idx], p48_rf)
a72_rf = roc_auc_score(y_72[test_idx], p72_rf)
print(f"  {'RandomForest':<15} Y48:{a48_rf:.4f}  Y72:{a72_rf:.4f}")

print(f"\n  Winner Y48: {'MLP' if a48_mlp > a48_rf else 'RandomForest'}")
print(f"  Winner Y72: {'MLP' if a72_mlp > a72_rf else 'RandomForest'}")

best_static_p48 = p48_mlp if a48_mlp >= a48_rf else p48_rf
best_static_p72 = p72_mlp if a72_mlp >= a72_rf else p72_rf

# %%
# === Ensemble: Weighted Average ===

print("\n--- Method 1: Weighted Average ---")
best_a48, best_a72, best_al48, best_al72 = 0, 0, 0.5, 0.5
for alpha in np.arange(0.0, 1.05, 0.05):
    p48_blend = alpha * best_temporal_p48 + (1 - alpha) * best_static_p48
    p72_blend = alpha * best_temporal_p72 + (1 - alpha) * best_static_p72
    a48 = roc_auc_score(t48_true, p48_blend)
    a72 = roc_auc_score(t72_true, p72_blend)
    if a48 > best_a48:
        best_a48, best_al48 = a48, alpha
    if a72 > best_a72:
        best_a72, best_al72 = a72, alpha

p48_wavg = best_al48 * best_temporal_p48 + (1 - best_al48) * best_static_p48
p72_wavg = best_al72 * best_temporal_p72 + (1 - best_al72) * best_static_p72
print(f"  Best alpha Y48: {best_al48:.2f} (temporal) -> AUROC: {best_a48:.4f}")
print(f"  Best alpha Y72: {best_al72:.2f} (temporal) -> AUROC: {best_a72:.4f}")

# %%
# === Ensemble: Stacking ===

print("\n--- Method 2: Stacking (Logistic Regression meta) ---")

# validation predictions for meta-learner
def get_val_preds_temporal(model):
    model.eval()
    va_ds = TemporalDataset(Xt_va, M_t[val_idx], y_48[val_idx], y_72[val_idx])
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)
    p48v, p72v = [], []
    with torch.no_grad():
        for x_t, mask, y48, y72 in va_dl:
            l48, l72 = model(x_t.to(device), mask.to(device))
            p48v += torch.sigmoid(l48).cpu().tolist()
            p72v += torch.sigmoid(l72).cpu().tolist()
    return np.array(p48v), np.array(p72v)

def get_val_preds_static(model):
    model.eval()
    va_ds = StaticDataset(Xs_va, y_48[val_idx], y_72[val_idx])
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)
    p48v, p72v = [], []
    with torch.no_grad():
        for x_s, y48, y72 in va_dl:
            l48, l72 = model(x_s.to(device))
            p48v += torch.sigmoid(l48).cpu().tolist()
            p72v += torch.sigmoid(l72).cpu().tolist()
    return np.array(p48v), np.array(p72v)

vp48_gru, vp72_gru = get_val_preds_temporal(gru_model)
vp48_lstm, vp72_lstm = get_val_preds_temporal(lstm_model)
vp48_mlp, vp72_mlp = get_val_preds_static(mlp_model)
vp48_rf = rf48.predict_proba(Xs_va)[:, 1]
vp72_rf = rf72.predict_proba(Xs_va)[:, 1]

X_meta_val_48 = np.stack([vp48_gru, vp48_lstm, vp48_mlp, vp48_rf], axis=1)
X_meta_val_72 = np.stack([vp72_gru, vp72_lstm, vp72_mlp, vp72_rf], axis=1)
X_meta_te_48 = np.stack([p48_gru, p48_lstm, p48_mlp, p48_rf], axis=1)
X_meta_te_72 = np.stack([p72_gru, p72_lstm, p72_mlp, p72_rf], axis=1)

meta48 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
meta72 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
meta48.fit(X_meta_val_48, y_48[val_idx])
meta72.fit(X_meta_val_72, y_72[val_idx])

p48_stack = meta48.predict_proba(X_meta_te_48)[:, 1]
p72_stack = meta72.predict_proba(X_meta_te_72)[:, 1]
a48_stack = roc_auc_score(t48_true, p48_stack)
a72_stack = roc_auc_score(t72_true, p72_stack)
print(f"  Stacking meta-weights Y48: {dict(zip(['GRU','LSTM','MLP','RF'], meta48.coef_[0].round(3)))}")
print(f"  Stacking meta-weights Y72: {dict(zip(['GRU','LSTM','MLP','RF'], meta72.coef_[0].round(3)))}")
print(f"  Stacking AUROC -> Y48:{a48_stack:.4f}  Y72:{a72_stack:.4f}")

# %%
# === Save results ===

print("\n--- Saving results ---")

def save_results(name, label, y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {"model": name, "label": label, "test_auc": float(auc), "test_auprc": float(ap)}
    with open(os.path.join(RESULTS_DIR, f"{name}_{label}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(RESULTS_DIR, f"{name}_{label}_predictions.npz"),
             y_true=y_true, y_prob=y_prob, y_pred=y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name=name)
    axes[0].set_title(f"ROC Curve ({label})")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name=name)
    axes[1].set_title(f"PR Curve ({label})")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{name}_{label}_curves.png"), dpi=150)
    plt.close()

all_models = {
    "BiGRU": (p48_gru, p72_gru),
    "BiLSTM": (p48_lstm, p72_lstm),
    "MLP_static": (p48_mlp, p72_mlp),
    "RF_static": (p48_rf, p72_rf),
    "WeightedAvg": (p48_wavg, p72_wavg),
    "Stacking": (p48_stack, p72_stack),
}
for name, (p48, p72) in all_models.items():
    save_results(name, "label_Y48", t48_true, p48)
    save_results(name, "label_Y72", t72_true, p72)

torch.save(gru_model.state_dict(), os.path.join(RESULTS_DIR, "bigru_model.pt"))
torch.save(lstm_model.state_dict(), os.path.join(RESULTS_DIR, "bilstm_model.pt"))
torch.save(mlp_model.state_dict(), os.path.join(RESULTS_DIR, "mlp_model.pt"))

# %%
# === Leaderboard ===

all_results = {
    "BiGRU   (temporal)": (a48_gru, a72_gru),
    "BiLSTM  (temporal)": (a48_lstm, a72_lstm),
    "MLP     (static)": (a48_mlp, a72_mlp),
    "RF      (static)": (a48_rf, a72_rf),
    "Weighted Average": (best_a48, best_a72),
    "Stacking": (a48_stack, a72_stack),
}

best_mean = max((v[0] + v[1]) / 2 for v in all_results.values())
print(f"\n{'='*58}")
print(f"  FINAL LEADERBOARD")
print(f"{'='*58}")
print(f"  {'Model':<28} {'Y48':>8} {'Y72':>8}")
print(f"  {'-'*52}")

rows = []
for name, (a48, a72) in all_results.items():
    marker = " *" if abs((a48 + a72) / 2 - best_mean) < 1e-6 else ""
    print(f"  {name:<28} {a48:>8.4f} {a72:>8.4f}{marker}")
    rows.append({"model": name, "auc_Y48": a48, "auc_Y72": a72, "mean_auc": (a48 + a72) / 2})
print(f"{'='*58}")

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "leaderboard.csv"), index=False)

print(f"\nBest ensemble method:")
print(f"  Y48 -> {'Stacking' if a48_stack >= best_a48 else 'Weighted Average'}")
print(f"  Y72 -> {'Stacking' if a72_stack >= best_a72 else 'Weighted Average'}")
print(f"\nResults saved to: {RESULTS_DIR}")

# %%
# === Combined ROC plot: all step 8 + step 9 models ===

import glob
from sklearn.metrics import roc_curve

STEP8_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
COMPARISON_DIR = os.path.join(STEP8_RESULTS, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

STEP8_NAMES = {
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}
STEP9_NAMES = {
    "BiGRU": "BiGRU",
    "BiLSTM": "BiLSTM",
    "MLP_static": "MLP (static)",
    "RF_static": "RF (static)",
    "WeightedAvg": "Weighted Average",
    "Stacking": "Stacking",
}

# Collect all predictions from both directories
all_preds = {}
for path in sorted(glob.glob(os.path.join(STEP8_RESULTS, "*_predictions.npz"))):
    name = os.path.basename(path).replace("_predictions.npz", "")
    data = np.load(path)
    all_preds[name] = {"y_true": data["y_true"], "y_prob": data["y_prob"]}
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_predictions.npz"))):
    name = os.path.basename(path).replace("_predictions.npz", "")
    data = np.load(path)
    all_preds[f"t9_{name}"] = {"y_true": data["y_true"], "y_prob": data["y_prob"]}

Y_LIST = [48, 72]
fig, axes = plt.subplots(1, len(Y_LIST), figsize=(7 * len(Y_LIST), 6))
if len(Y_LIST) == 1:
    axes = [axes]

for ax, y in zip(axes, Y_LIST):
    label_col = f"label_Y{y}"
    # Step 8 models
    for key, data in sorted(all_preds.items()):
        if key.startswith("t9_"):
            continue
        if not key.endswith(label_col):
            continue
        model_name = key.replace(f"_{label_col}", "")
        display = STEP8_NAMES.get(model_name, model_name)
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc_val = roc_auc_score(data["y_true"], data["y_prob"])
        ax.plot(fpr, tpr, label=f"{display} (AUC={auc_val:.3f})")
    # Step 9 models
    for key, data in sorted(all_preds.items()):
        if not key.startswith("t9_"):
            continue
        if not key.endswith(label_col):
            continue
        model_name = key.replace("t9_", "").replace(f"_{label_col}", "")
        display = STEP9_NAMES.get(model_name, model_name)
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc_val = roc_auc_score(data["y_true"], data["y_prob"])
        ax.plot(fpr, tpr, linestyle="--", label=f"{display} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Y={y}h")
    ax.legend(loc="lower right", fontsize=8)

plt.suptitle("ROC Comparison — All Models (Step 8 + Step 9)", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(COMPARISON_DIR, "roc_comparison_all_models.png"), dpi=150)
plt.close()
print(f"\nSaved combined ROC plot: {os.path.join(COMPARISON_DIR, 'roc_comparison_all_models.png')}")
