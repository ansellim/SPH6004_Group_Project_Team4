# %%
# Step 5: Advanced Time Series Features
import pandas as pd
import numpy as np

# %%
# load time series window from step 4
df_ts = pd.read_csv("ts_window.csv")
all_stay_ids = sorted(df_ts["stay_id"].unique())
print(f"Time series: {len(df_ts)} rows, {len(all_stay_ids)} stays")

# %%
# === Last 6h measurement counts (18-24h window) ===
LAST_6H_COUNT_VARS = [
    "map", "vasopressor_dose", "gcs", "sofa_cns", "sofa_cardio",
    "lactate", "sodium", "sofa_renal", "sofa_resp",
]

df_last6h = df_ts[df_ts["relative_hour"] > 18].copy()
df_counts = pd.DataFrame({"stay_id": all_stay_ids})

for var in LAST_6H_COUNT_VARS:
    counts = df_last6h.dropna(subset=[var]).groupby("stay_id").size()
    df_counts[f"{var}_count_last6h"] = counts.reindex(all_stay_ids, fill_value=0).values

print(f"Last 6h count features: {len(LAST_6H_COUNT_VARS)}")

# %%
# === Time to first abnormal ===
FIRST_ABNORMAL = {
    "rr":   {"lower": 12, "upper": 20},
    "temp": {"lower": 36.0, "upper": 38.0},
    "map":  {"lower": 65, "upper": 110},
    "hr":   {"lower": 60, "upper": 100},
}

df_abnormal = pd.DataFrame({"stay_id": all_stay_ids})
for var, thresholds in FIRST_ABNORMAL.items():
    sub = df_ts[["stay_id", "relative_hour", var]].dropna(subset=[var])
    mask = (sub[var] < thresholds["lower"]) | (sub[var] > thresholds["upper"])
    first = sub[mask].groupby("stay_id")["relative_hour"].min()
    df_abnormal[f"{var}_time_to_first_abnormal_hours"] = first.reindex(all_stay_ids).values

print(f"Time-to-first-abnormal features: {len(FIRST_ABNORMAL)}")

# %%
# === Time to worst (peak or trough) ===
NADIR_CONFIG = {
    "creatinine": "high", "hr": "high", "rr": "high", "temp": "high",
    "platelets": "low", "bun": "high", "sodium": "low",
}

df_nadir = pd.DataFrame({"stay_id": all_stay_ids})
for var, worst in NADIR_CONFIG.items():
    sub = df_ts[["stay_id", "relative_hour", var]].dropna(subset=[var])
    if sub.empty:
        df_nadir[f"{var}_time_to_{'peak' if worst == 'high' else 'trough'}_hours"] = np.nan
        continue
    if worst == "high":
        idx = sub.groupby("stay_id")[var].idxmax()
    else:
        idx = sub.groupby("stay_id")[var].idxmin()
    hours = sub.loc[idx].set_index("stay_id")["relative_hour"]
    col = f"{var}_time_to_{'peak' if worst == 'high' else 'trough'}_hours"
    df_nadir[col] = hours.reindex(all_stay_ids).values

print(f"Time-to-worst features: {len(NADIR_CONFIG)}")

# %%
# === Exposure time above upper threshold (LOCF) ===
EXPOSURE_UPPER = {"rr": 20, "hr": 100, "bun": 20}

df_exposure = pd.DataFrame({"stay_id": all_stay_ids})
for var, upper in EXPOSURE_UPPER.items():
    sub = df_ts[["stay_id", "relative_hour", var]].dropna(subset=[var])
    sub = sub.sort_values(["stay_id", "relative_hour"]).copy()
    next_hour = sub.groupby("stay_id")["relative_hour"].shift(-1)
    sub["duration"] = (next_hour.fillna(24) - sub["relative_hour"]).clip(lower=0)
    sub["above_dur"] = sub["duration"] * (sub[var] > upper).astype(float)
    total = sub.groupby("stay_id")["above_dur"].sum()
    df_exposure[f"{var}_time_above_upper_hours"] = total.reindex(all_stay_ids, fill_value=0).values

print(f"Exposure features: {len(EXPOSURE_UPPER)}")

# %%
# merge all advanced features
df_advanced = df_counts.merge(df_abnormal, on="stay_id")
df_advanced = df_advanced.merge(df_nadir, on="stay_id")
df_advanced = df_advanced.merge(df_exposure, on="stay_id")
df_advanced.to_csv("advanced_features.csv", index=False)
print(f"Saved advanced_features.csv: {df_advanced.shape}")
