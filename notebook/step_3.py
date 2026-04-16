# %%
# Step 3: Static Features (demographics + static labs)
import pandas as pd

# %%
# load cohort from step 2
df = pd.read_csv("cohort_step_2.csv")
print(f"Cohort: {len(df)} patients, {len(df.columns)} columns")

# %%
# define static-only lab columns (26 labs)
STATIC_ONLY_LABS = [
    "sbp_min", "sbp_max", "sbp_mean", "dbp_mean",
    "glucose_min",
    "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable",
    "hematocrit_min", "hematocrit_max", "hemoglobin_min", "hemoglobin_max",
    "aniongap_min", "aniongap_max", "bicarbonate_min",
    "calcium_min", "calcium_max", "chloride_min", "chloride_max",
    "potassium_max",
    "thrombin_min", "thrombin_max",
    "pt_min", "ptt_min", "ptt_max",
]

DEMOGRAPHIC_COLS = ["age", "gender", "insurance", "race", "marital_status", "first_careunit"]

# %%
# keep only columns that exist in the dataframe
available_labs = [c for c in STATIC_ONLY_LABS if c in df.columns]
available_demo = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
print(f"Available: {len(available_demo)} demographics, {len(available_labs)} static labs")

# %%
# build features dataframe
features = df[["subject_id", "stay_id"] + available_demo + available_labs + ["language"]].copy()

# rename labs with _static suffix
features.rename(columns={c: f"{c}_static" for c in available_labs}, inplace=True)

# %%
# race grouping: 34 MIMIC categories -> 6 groups
RACE_MAP = {
    "WHITE": "White", "WHITE - OTHER EUROPEAN": "White",
    "WHITE - RUSSIAN": "White", "WHITE - EASTERN EUROPEAN": "White",
    "WHITE - BRAZILIAN": "White", "PORTUGUESE": "White",
    "BLACK/AFRICAN AMERICAN": "Black", "BLACK/CAPE VERDEAN": "Black",
    "BLACK/CARIBBEAN ISLAND": "Black", "BLACK/AFRICAN": "Black",
    "ASIAN": "Asian", "ASIAN - CHINESE": "Asian",
    "ASIAN - SOUTH EAST ASIAN": "Asian", "ASIAN - ASIAN INDIAN": "Asian",
    "ASIAN - KOREAN": "Asian",
    "HISPANIC OR LATINO": "Hispanic", "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic", "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - SALVADORAN": "Hispanic", "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "HISPANIC/LATINO - COLUMBIAN": "Hispanic", "HISPANIC/LATINO - CUBAN": "Hispanic",
    "HISPANIC/LATINO - HONDURAN": "Hispanic", "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    "SOUTH AMERICAN": "Hispanic",
    "OTHER": "Other", "AMERICAN INDIAN/ALASKA NATIVE": "Other",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Other",
    "MULTIPLE RACE/ETHNICITY": "Other",
    "UNKNOWN": "Unknown", "UNABLE TO OBTAIN": "Unknown",
    "PATIENT DECLINED TO ANSWER": "Unknown",
}
features["race"] = features["race"].map(RACE_MAP).fillna("Other")
print(f"Race groups: {features['race'].value_counts().to_dict()}")

# %%
# marital status grouping: 4 -> 3
MARITAL_MAP = {
    "MARRIED": "Married",
    "SINGLE": "Single",
    "WIDOWED": "Previously_Married",
    "DIVORCED": "Previously_Married",
}
features["marital_status"] = features["marital_status"].map(MARITAL_MAP)

# %%
# language -> english_speaker binary
features["english_speaker"] = (features["language"] == "ENGLISH").astype(int)
features.drop(columns=["language"], inplace=True)

# %%
# save static features
features.to_csv("static_features.csv", index=False)
print(f"Saved static_features.csv: {features.shape}")
