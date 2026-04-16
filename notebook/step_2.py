# %%
# Step 2: Label Construction + Train/Test Split
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
# load cohort from step 1
df = pd.read_csv("cohort.csv")
print(f"Cohort: {len(df)} patients")

# %%
# create label_Y48: discharged alive within 48 hours
df["label_Y48"] = ((df["icu_death_flag"] == 0) & (df["icu_los_hours"] <= 48)).astype(int)
print(f"Y48: {df['label_Y48'].sum()} positive ({df['label_Y48'].mean()*100:.1f}%)")

# %%
# create label_Y72: discharged alive within 72 hours
df["label_Y72"] = ((df["icu_death_flag"] == 0) & (df["icu_los_hours"] <= 72)).astype(int)
print(f"Y72: {df['label_Y72'].sum()} positive ({df['label_Y72'].mean()*100:.1f}%)")

# %%
# train/test split: 70/30 stratified by label_Y48, at patient level
train_ids, test_ids = train_test_split(
    df["subject_id"].unique(),
    test_size=0.30,
    random_state=42,
    stratify=df.groupby("subject_id")["label_Y48"].first(),
)
df["split"] = df["subject_id"].apply(lambda x: "train" if x in set(train_ids) else "test")

train_n = (df["split"] == "train").sum()
test_n = (df["split"] == "test").sum()
print(f"Train: {train_n} ({train_n/len(df)*100:.1f}%), Test: {test_n} ({test_n/len(df)*100:.1f}%)")

# %%
# save cohort with labels and split
df.to_csv("cohort_step_2.csv", index=False)
print(f"Saved cohort_step_2.csv: {df.shape}")
