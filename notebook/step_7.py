# %%
# Step 7: Merge All Features + Data Preparation
import pandas as pd
import numpy as np

# %%
# === Part A: Merge all feature files ===

# load cohort with labels and split
cohort = pd.read_csv("cohort_step_2.csv")
keep_cols = ["subject_id", "stay_id", "label_Y48", "label_Y72", "split"]
df = cohort[keep_cols].copy()
print(f"Cohort: {len(df)}")

# merge static features
static = pd.read_csv("static_features.csv")
df = df.merge(static, on=["subject_id", "stay_id"], how="left")
print(f"After static: {df.shape}")

# merge time series aggregation
ts_agg = pd.read_csv("ts_agg.csv")
df = df.merge(ts_agg, on="stay_id", how="left")
print(f"After ts_agg: {df.shape}")

# merge text features
text = pd.read_csv("text_features.csv")
df = df.merge(text, on="stay_id", how="left")
print(f"After text: {df.shape}")

# merge advanced features
advanced = pd.read_csv("advanced_features.csv")
df = df.merge(advanced, on="stay_id", how="left")
print(f"After advanced: {df.shape}")

# %%
# save pre-imputation merged dataset
df.to_csv("final_dataset.csv", index=False)
print(f"Saved final_dataset.csv: {df.shape}")

# %%
# === Part B: Feature selection and encoding ===

NON_FEATURE_COLS = ["subject_id", "stay_id", "label_Y48", "label_Y72", "split"]

train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
all_features = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"Train: {len(train)}, Test: {len(test)}, Features: {len(all_features)}")

# %%
# separate numeric and categorical, drop high-missingness columns
cat_cols = [c for c in all_features if not pd.api.types.is_numeric_dtype(train[c])]
num_cols = [c for c in all_features if pd.api.types.is_numeric_dtype(train[c])]

MISS_THRESHOLD = 0.20
miss_rate = train[num_cols].isna().mean()
keep_num = miss_rate[miss_rate < MISS_THRESHOLD].index.tolist()
dropped = miss_rate[miss_rate >= MISS_THRESHOLD].index.tolist()
keep_cat = [c for c in cat_cols if train[c].isna().mean() < MISS_THRESHOLD]

print(f"Numeric: {len(keep_num)} kept, {len(dropped)} dropped (>={MISS_THRESHOLD*100:.0f}% missing)")
print(f"Categorical: {len(keep_cat)} kept")

# %%
# one-hot encode categoricals
if keep_cat:
    train_enc = pd.get_dummies(train[keep_cat], columns=keep_cat, dummy_na=True)
    test_enc = pd.get_dummies(test[keep_cat], columns=keep_cat, dummy_na=True)
    for c in train_enc.columns:
        if c not in test_enc.columns:
            test_enc[c] = 0
    for c in test_enc.columns:
        if c not in train_enc.columns:
            train_enc[c] = 0
    test_enc = test_enc[train_enc.columns]
    print(f"One-hot columns: {len(train_enc.columns)}")
else:
    train_enc = pd.DataFrame(index=train.index)
    test_enc = pd.DataFrame(index=test.index)

# %%
# === Part C: MICE imputation ===
import miceforest as mf

train_num = train[keep_num]
test_num = test[keep_num]
cols_with_na = [c for c in train_num.columns if train_num[c].isna().any()]
print(f"Columns needing imputation: {len(cols_with_na)}, already complete: {len(keep_num) - len(cols_with_na)}")

# %%
# run MICE on train, impute test
if cols_with_na:
    kernel = mf.ImputationKernel(
        train_num, num_datasets=1, variable_schema=cols_with_na,
        random_state=42,
    )
    kernel.mice(5, verbose=True, num_threads=4)
    test_new = kernel.impute_new_data(test_num)
    train_num = kernel.complete_data(dataset=0)
    test_num = test_new.complete_data(dataset=0)

# %%
# === Part D: Assemble and save ===
id_label_cols = ["subject_id", "stay_id", "label_Y48", "label_Y72"]

train_final = pd.concat([
    train[id_label_cols].reset_index(drop=True),
    train_enc.reset_index(drop=True),
    train_num.reset_index(drop=True),
], axis=1)
train_final["split"] = "train"

test_final = pd.concat([
    test[id_label_cols].reset_index(drop=True),
    test_enc.reset_index(drop=True),
    test_num.reset_index(drop=True),
], axis=1)
test_final["split"] = "test"

result = pd.concat([train_final, test_final], ignore_index=True)

# %%
# save final imputed dataset and feature list
result.to_csv("final_dataset_imputed.csv", index=False)
feature_cols = [c for c in result.columns if c not in NON_FEATURE_COLS]
pd.DataFrame({"feature": feature_cols}).to_csv("feature_list_imputed.csv", index=False)

nans = result[feature_cols].isna().sum().sum()
print(f"Saved final_dataset_imputed.csv: {result.shape}")
print(f"Features: {len(feature_cols)}, NaNs: {nans}")
