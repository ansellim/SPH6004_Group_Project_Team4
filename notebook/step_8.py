# %%
# Step 8: Train models + Evaluate
# Combines logic from pipeline/train_models.py and pipeline/evaluate.py

import matplotlib
matplotlib.use("Agg")

import os
import sys
import json
import time
import copy
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    roc_curve, precision_recall_curve,
)
from scipy.stats import loguniform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42
N_SEARCH = 20
CV_FOLDS = 5
Y_LIST = [48, 72]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

NON_FEATURE_COLS = ["subject_id", "stay_id", "label_Y48", "label_Y72", "split"]

# %%
# ============================================================
# Part A: Load data and setup
# ============================================================

print("=== Part A: Load data ===\n")

df = pd.read_csv("final_dataset_imputed.csv")
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
train_X = train_df[feature_cols]
test_X = test_df[feature_cols]

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Features: {len(feature_cols)}")


def standardize(tr, te):
    """Fit StandardScaler on train, transform both."""
    scaler = StandardScaler()
    cols = tr.columns
    tr_s = pd.DataFrame(scaler.fit_transform(tr), columns=cols)
    te_s = pd.DataFrame(scaler.transform(te), columns=cols)
    return tr_s, te_s, scaler


# %%
# ============================================================
# Part B: Evaluation helper
# ============================================================

def evaluate_and_save(model_name, label_col, y_test, y_prob, y_pred,
                      best_params, cv_score, feat_cols=None, feat_imp=None):
    """Compute metrics, save JSON/npz/plots."""
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    print(f"  Test AUC:   {auc:.4f}")
    print(f"  Test AUPRC: {ap:.4f}")
    print(f"\n  Classification report:\n{classification_report(y_test, y_pred)}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save metrics JSON
    metrics = {
        "model": model_name, "label": label_col,
        "cv_auc": float(cv_score), "test_auc": float(auc), "test_auprc": float(ap),
        "best_params": best_params,
    }
    with open(os.path.join(RESULTS_DIR, f"{model_name}_{label_col}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save predictions
    np.savez(
        os.path.join(RESULTS_DIR, f"{model_name}_{label_col}_predictions.npz"),
        y_true=np.array(y_test), y_prob=np.array(y_prob), y_pred=np.array(y_pred),
    )

    # ROC and PR curve plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name=model_name)
    axes[0].set_title(f"ROC Curve ({label_col})")
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=model_name)
    axes[1].set_title(f"PR Curve ({label_col})")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{model_name}_{label_col}_curves.png"), dpi=150)
    plt.close()

    # Feature importance plot (top 20)
    if feat_imp is not None and feat_cols is not None:
        imp = pd.Series(feat_imp, index=feat_cols).sort_values(ascending=False)
        imp.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_{label_col}_feature_importance.csv"))
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.head(20).plot.barh(ax=ax)
        ax.set_title(f"Top 20 Features ({model_name}, {label_col})")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"{model_name}_{label_col}_importance.png"), dpi=150)
        plt.close()

    return metrics


# %%
# ============================================================
# Part C: Train 4 sklearn models
# ============================================================

print("\n=== Part C: Train sklearn models ===\n")

for y in Y_LIST:
    label_col = f"label_Y{y}"
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    # ----------------------------------------------------------
    # 11. Decision Tree
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Decision Tree - {label_col}")
    print(f"{'='*60}")
    t0 = time.time()

    dt_params = {
        "max_depth": [3, 5, 7, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf": [1, 5, 10, 20, 50],
        "max_features": ["sqrt", "log2", 0.5, 0.8, None],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        param_distributions=dt_params,
        n_iter=N_SEARCH, cv=CV_FOLDS, scoring="roc_auc",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=2,
    )
    search.fit(train_X, y_train)

    best = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")

    y_prob = best.predict_proba(test_X)[:, 1]
    y_pred = best.predict(test_X)

    evaluate_and_save(
        "decision_tree", label_col, y_test, y_prob, y_pred,
        best_params=search.best_params_, cv_score=search.best_score_,
        feat_cols=feature_cols, feat_imp=best.feature_importances_,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # ----------------------------------------------------------
    # 12. Logistic Regression (needs scaling)
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Logistic Regression - {label_col}")
    print(f"{'='*60}")
    t0 = time.time()

    train_X_s, test_X_s, _ = standardize(train_X, test_X)

    lr_params = {
        "C": loguniform(1e-3, 1e2),
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        LogisticRegression(solver="saga", max_iter=2000, random_state=RANDOM_SEED),
        param_distributions=lr_params,
        n_iter=N_SEARCH, cv=CV_FOLDS, scoring="roc_auc",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=2,
    )
    search.fit(train_X_s, y_train)

    best = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")

    y_prob = best.predict_proba(test_X_s)[:, 1]
    y_pred = best.predict(test_X_s)

    evaluate_and_save(
        "logistic_regression", label_col, y_test, y_prob, y_pred,
        best_params=search.best_params_, cv_score=search.best_score_,
        feat_cols=feature_cols, feat_imp=np.abs(best.coef_[0]),
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # ----------------------------------------------------------
    # 13. Random Forest
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Random Forest - {label_col}")
    print(f"{'='*60}")
    t0 = time.time()

    rf_params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        param_distributions=rf_params,
        n_iter=N_SEARCH, cv=CV_FOLDS, scoring="roc_auc",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=2,
    )
    search.fit(train_X, y_train)

    best = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")

    y_prob = best.predict_proba(test_X)[:, 1]
    y_pred = best.predict(test_X)

    evaluate_and_save(
        "random_forest", label_col, y_test, y_prob, y_pred,
        best_params=search.best_params_, cv_score=search.best_score_,
        feat_cols=feature_cols, feat_imp=best.feature_importances_,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # ----------------------------------------------------------
    # 14. XGBoost
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  XGBoost - {label_col}")
    print(f"{'='*60}")
    t0 = time.time()

    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    xgb_params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.1, 0.5, 1.0],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    search = RandomizedSearchCV(
        XGBClassifier(
            eval_metric="logloss", tree_method="hist", n_jobs=4,
            random_state=RANDOM_SEED, scale_pos_weight=scale_pos_weight,
        ),
        param_distributions=xgb_params,
        n_iter=N_SEARCH, cv=CV_FOLDS, scoring="roc_auc",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=2,
    )
    search.fit(train_X, y_train)

    best = search.best_estimator_
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")

    y_prob = best.predict_proba(test_X)[:, 1]
    y_pred = best.predict(test_X)

    evaluate_and_save(
        "xgboost", label_col, y_test, y_prob, y_pred,
        best_params=search.best_params_, cv_score=search.best_score_,
        feat_cols=feature_cols, feat_imp=best.feature_importances_,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

print("\nAll sklearn models trained.")

# %%
# ============================================================
# Part D: Neural Network (PyTorch MLP)
# ============================================================

print("\n=== Part D: Neural Network ===\n")

# 15. Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# 16. MLP architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout, batch_norm=True):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_one_fold(cfg, X_train, y_train, X_val, y_val, input_dim):
    """Train one fold, return best val AUC and model state dict."""
    model = MLP(input_dim, cfg["hidden_layers"], cfg["dropout"],
                cfg.get("batch_norm", True)).to(DEVICE)

    pos_weight = torch.tensor(
        [(1 - y_train.mean()) / y_train.mean()], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    train_loader = make_loader(X_train, y_train, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg["batch_size"], shuffle=False)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                val_probs.append(torch.sigmoid(model(X_b.to(DEVICE))).cpu().numpy())
        val_auc = roc_auc_score(y_val, np.concatenate(val_probs).ravel())

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                break

    return best_val_auc, best_state


def nn_predict(model, X, batch_size=512):
    """Get predicted probabilities from trained model."""
    model.eval()
    loader = make_loader(X, np.zeros(len(X)), batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for X_b, _ in loader:
            probs.append(torch.sigmoid(model(X_b.to(DEVICE))).cpu().numpy())
    return np.concatenate(probs).ravel()


# %%
# 17. Search space
NN_SEARCH_SPACE = {
    "hidden_layers": [
        [256, 128, 64, 32], [128, 64, 32, 16], [64, 32, 16, 8],
        [256, 128, 64], [128, 64, 32], [64, 32, 16],
        [256, 128], [128, 64],
    ],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "weight_decay": [0, 1e-5, 1e-4, 1e-3],
    "batch_size": [128, 256, 512],
}

# 18. Default config
NN_DEFAULT = {
    "hidden_layers": [128, 64, 32, 16],
    "dropout": 0.3,
    "batch_norm": True,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 500,
    "patience": 30,
}

# %%
# 19-20. Train neural network for each Y label

rng = np.random.default_rng(RANDOM_SEED)

# Standardize for NN
train_X_s, test_X_s, _ = standardize(train_X, test_X)
X_train_np = train_X_s.values.astype(np.float32)
X_test_np = test_X_s.values.astype(np.float32)
input_dim = X_train_np.shape[1]

for y in Y_LIST:
    label_col = f"label_Y{y}"
    y_train = train_df[label_col].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    print(f"\n{'='*60}")
    print(f"  Neural Network - {label_col}")
    print(f"{'='*60}")
    t0 = time.time()

    # 19. Random search: 20 configs x 5-fold CV
    best_cv_auc = 0
    best_cfg = None

    for i in range(N_SEARCH):
        cfg = copy.deepcopy(NN_DEFAULT)
        for key, values in NN_SEARCH_SPACE.items():
            cfg[key] = values[rng.integers(len(values))]

        layers_str = "-".join(map(str, cfg["hidden_layers"]))
        print(f"  Config {i+1}/{N_SEARCH}: layers={layers_str}, "
              f"lr={cfg['learning_rate']}, drop={cfg['dropout']}, bs={cfg['batch_size']}")

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_aucs = []
        for train_idx, val_idx in skf.split(X_train_np, y_train):
            auc_val, _ = train_one_fold(
                cfg, X_train_np[train_idx], y_train[train_idx],
                X_train_np[val_idx], y_train[val_idx], input_dim,
            )
            fold_aucs.append(auc_val)
        cv_auc = np.mean(fold_aucs)
        print(f"    CV AUC: {cv_auc:.4f}")

        if cv_auc > best_cv_auc:
            best_cv_auc = cv_auc
            best_cfg = cfg

    print(f"\n  Best config: layers={'-'.join(map(str, best_cfg['hidden_layers']))}, "
          f"lr={best_cfg['learning_rate']}, CV AUC={best_cv_auc:.4f}")

    # 20. Retrain best config on 90/10 split, predict on test
    print("  Retraining on 90/10 split...")
    n = len(X_train_np)
    rng_np = np.random.RandomState(RANDOM_SEED)
    idx = rng_np.permutation(n)
    split_point = int(n * 0.9)

    _, best_state = train_one_fold(
        best_cfg,
        X_train_np[idx[:split_point]], y_train[idx[:split_point]],
        X_train_np[idx[split_point:]], y_train[idx[split_point:]],
        input_dim,
    )

    model = MLP(input_dim, best_cfg["hidden_layers"], best_cfg["dropout"],
                best_cfg.get("batch_norm", True)).to(DEVICE)
    model.load_state_dict(best_state)

    y_prob = nn_predict(model, X_test_np)
    y_pred = (y_prob >= 0.5).astype(int)

    evaluate_and_save(
        "neural_network", label_col, y_test, y_prob, y_pred,
        best_params={k: v for k, v in best_cfg.items() if k != "max_epochs"},
        cv_score=best_cv_auc, feat_cols=feature_cols,
    )

    # Save model and config
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"neural_network_{label_col}.pt"))
    with open(os.path.join(RESULTS_DIR, f"neural_network_{label_col}_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    print(f"  Time: {time.time() - t0:.1f}s")

print("\nAll neural network models trained.")

# %%
# ============================================================
# Part E: Cross-model comparison
# ============================================================

print("\n=== Part E: Cross-model comparison ===\n")

DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}

# 21. Collect all metrics JSON files
rows = []
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_metrics.json"))):
    with open(path) as f:
        m = json.load(f)
    rows.append({
        "model": m["model"], "label": m["label"],
        "cv_auc": m["cv_auc"], "test_auc": m["test_auc"], "test_auprc": m["test_auprc"],
    })
metrics_df = pd.DataFrame(rows)

if metrics_df.empty:
    print("No metrics found. Something went wrong.")
else:
    # 22. Print comparison table sorted by test_auc
    for y in Y_LIST:
        label_col = f"label_Y{y}"
        sub = metrics_df[metrics_df["label"] == label_col].sort_values("test_auc", ascending=False)
        print(f"\n  {label_col}:")
        print(sub[["model", "cv_auc", "test_auc", "test_auprc"]].to_string(index=False))

    # Collect predictions
    preds = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_predictions.npz"))):
        name = os.path.basename(path).replace("_predictions.npz", "")
        data = np.load(path)
        preds[name] = {"y_true": data["y_true"], "y_prob": data["y_prob"]}

    # 23. Overlaid ROC curves for each Y label
    for y in Y_LIST:
        label_col = f"label_Y{y}"
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, data in sorted(preds.items()):
            if not key.endswith(label_col):
                continue
            model_name = key.replace(f"_{label_col}", "")
            display = DISPLAY_NAMES.get(model_name, model_name)
            fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
            auc_val = roc_auc_score(data["y_true"], data["y_prob"])
            ax.plot(fpr, tpr, label=f"{display} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Comparison ({label_col})")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(COMPARISON_DIR, f"roc_comparison_{label_col}.png"), dpi=150)
        plt.close()
        print(f"  Saved ROC comparison for {label_col}")

    # 24. Overlaid PR curves for each Y label
    for y in Y_LIST:
        label_col = f"label_Y{y}"
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, data in sorted(preds.items()):
            if not key.endswith(label_col):
                continue
            model_name = key.replace(f"_{label_col}", "")
            display = DISPLAY_NAMES.get(model_name, model_name)
            prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
            ap = average_precision_score(data["y_true"], data["y_prob"])
            ax.plot(rec, prec, label=f"{display} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Comparison ({label_col})")
        ax.legend(loc="lower left")
        plt.tight_layout()
        fig.savefig(os.path.join(COMPARISON_DIR, f"pr_comparison_{label_col}.png"), dpi=150)
        plt.close()
        print(f"  Saved PR comparison for {label_col}")

    # 25. Bar charts for test_auc, test_auprc, cv_auc
    for metric in ["test_auc", "test_auprc", "cv_auc"]:
        fig, axes = plt.subplots(1, len(Y_LIST), figsize=(6 * len(Y_LIST), 5))
        if len(Y_LIST) == 1:
            axes = [axes]
        for ax, y in zip(axes, Y_LIST):
            label_col = f"label_Y{y}"
            sub = metrics_df[metrics_df["label"] == label_col].sort_values(metric, ascending=True)
            display_names = [DISPLAY_NAMES.get(m, m) for m in sub["model"]]
            ax.barh(display_names, sub[metric])
            ax.set_xlabel(metric.replace("_", " ").upper())
            ax.set_title(f"{label_col}")
            for i, v in enumerate(sub[metric]):
                ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
        plt.suptitle(metric.replace("_", " ").upper(), fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(COMPARISON_DIR, f"{metric}_comparison.png"), dpi=150)
        plt.close()
    print("  Saved metric bar charts")

    # 26. Save comparison CSV
    metrics_df.to_csv(os.path.join(COMPARISON_DIR, "model_comparison.csv"), index=False)
    print(f"  Saved: {os.path.join(COMPARISON_DIR, 'model_comparison.csv')}")

# %%
print("\n=== Step 8 complete ===")
