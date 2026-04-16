# %%
# Step 4: Time Series Preparation + Aggregation
import pandas as pd
import numpy as np

# %%
# load cohort intime for relative hour calculation
df_cohort = pd.read_csv("cohort_step_2.csv", usecols=["stay_id", "intime"])
df_cohort["intime"] = pd.to_datetime(df_cohort["intime"])
cohort_stay_ids = set(df_cohort["stay_id"])
print(f"Cohort stays: {len(cohort_stay_ids)}")

# %%
# read time series in chunks, filter to cohort
chunks = []
for chunk in pd.read_csv("../data/MIMIC-IV-time_series(Group Assignment).csv",
                         chunksize=500_000, na_values="NULL"):
    chunk = chunk[chunk["stay_id"].isin(cohort_stay_ids)]
    chunks.append(chunk)
df_ts = pd.concat(chunks, ignore_index=True)
del chunks
print(f"Time series rows: {len(df_ts)}")

# %%
# compute relative hour and clean up
df_ts["hour_ts"] = pd.to_datetime(df_ts["hour_ts"])
intime_map = df_cohort.set_index("stay_id")["intime"]
df_ts["intime"] = df_ts["stay_id"].map(intime_map)
df_ts["relative_hour"] = (df_ts["hour_ts"] - df_ts["intime"]).dt.total_seconds() / 3600
df_ts.drop(columns=["hour_ts", "intime"], inplace=True)

# drop spo2 and inr (100% missing)
df_ts.drop(columns=["spo2", "inr"], errors="ignore", inplace=True)

# %%
# save full time series with relative hours
df_ts.to_csv("timeseries_relative.csv", index=False)
print(f"Saved timeseries_relative.csv: {df_ts.shape}")

# %%
# filter to observation window [-3, 24] hours
df_ts_window = df_ts[(df_ts["relative_hour"] >= -3) & (df_ts["relative_hour"] <= 24)].copy()
df_ts_window.to_csv("ts_window.csv", index=False)
print(f"Observation window: {len(df_ts_window)} rows, {df_ts_window['stay_id'].nunique()} stays")

# %%
# define variable groups (matching pipeline config)
DENSE_VARS = ["hr", "rr", "ventilation_flag", "urine_output", "sofa_cardio"]
SPARSE_VARS = [
    "gcs", "temp", "sofa_cns", "sofa_renal", "sodium", "creatinine", "bun",
    "platelets", "vasopressor_dose", "sofa_coag", "map", "lactate", "sofa_resp",
    "bilirubin", "sofa_liver", "fluid_input", "wbc",
]

def slope(series):
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid))
    return np.polyfit(x, valid.values, 1)[0]

# %%
# aggregate dense variables: mean, last, std, slope
agg_frames = []
for var in DENSE_VARS:
    grp = df_ts_window.groupby("stay_id")[var]
    agg = grp.agg(mean="mean", last="last", std="std")
    agg["slope"] = grp.apply(slope)
    agg.columns = [f"{var}_{s}" for s in agg.columns]
    agg_frames.append(agg)
print(f"Dense vars: {len(DENSE_VARS)} x 4 stats = {len(DENSE_VARS)*4} features")

# %%
# aggregate GCS: mean, last, min, max (special case)
grp = df_ts_window.groupby("stay_id")["gcs"]
gcs_agg = grp.agg(mean="mean", last="last", **{"min": "min", "max": "max"})
gcs_agg.columns = [f"gcs_{s}" for s in gcs_agg.columns]
agg_frames.append(gcs_agg)
print("GCS: 4 features (mean, last, min, max)")

# %%
# aggregate other sparse variables: mean, last only
other_sparse = [v for v in SPARSE_VARS if v != "gcs"]
for var in other_sparse:
    grp = df_ts_window.groupby("stay_id")[var]
    agg = grp.agg(mean="mean", last="last")
    agg.columns = [f"{var}_{s}" for s in agg.columns]
    agg_frames.append(agg)
print(f"Other sparse vars: {len(other_sparse)} x 2 stats = {len(other_sparse)*2} features")

# %%
# merge all aggregations
df_ts_agg = pd.concat(agg_frames, axis=1).reset_index()
df_ts_agg.to_csv("ts_agg.csv", index=False)
print(f"Saved ts_agg.csv: {df_ts_agg.shape}")
