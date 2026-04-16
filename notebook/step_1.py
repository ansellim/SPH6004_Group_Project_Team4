# %%
# Step 1: Cohort Selection
import pandas as pd

# %%
# load raw static data
df_static = pd.read_csv("../data/MIMIC-IV-static(Group Assignment).csv")
print(f"Raw data: {df_static.shape}")

# %%
# exclude non-ICU units
EXCLUDED_CAREUNITS = ["Neuro Intermediate", "Neuro Stepdown"]
df = df_static[~df_static["first_careunit"].isin(EXCLUDED_CAREUNITS)]
print(f"After excluding non-ICU units: {len(df)} ({len(df_static) - len(df)} removed)")

# %%
# keep first ICU stay per patient (by intime)
df["intime"] = pd.to_datetime(df["intime"])
df = df.sort_values("intime").groupby("subject_id").first().reset_index()
print(f"After keeping first stay per patient: {len(df)}")

# %%
# require ICU LOS > 24 hours
df = df[df["icu_los_hours"] > 24]
print(f"After requiring LOS > 24h: {len(df)}")

# %%
# exclude patients who died before 24 hours
df["deathtime"] = pd.to_datetime(df["deathtime"])
hours_to_death = (df["deathtime"] - df["intime"]).dt.total_seconds() / 3600
died_before_24 = hours_to_death.lt(24).fillna(False)
df = df[~died_before_24]
print(f"After excluding deaths before 24h: {len(df)}")

# %%
# save cohort
df.to_csv("cohort.csv", index=False)
print(f"Saved cohort.csv: {df.shape}")
