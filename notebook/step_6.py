# %%
# import relevant packages
import pandas as pd
import numpy as np
import re

# %%
# load cohort and text data
df_cohort = pd.read_csv("cohort_step_2.csv", usecols=["stay_id", "intime"])
df_cohort["intime"] = pd.to_datetime(df_cohort["intime"])
print(f"Cohort: {len(df_cohort)} stays")

df_text = pd.read_csv("../data/MIMIC-IV-text(Group Assignment).csv")
print(f"Raw text data: {len(df_text)} rows")

# %%
# filter text to cohort stay_ids
cohort_stay_ids = set(df_cohort["stay_id"])
df_text = df_text[df_text["stay_id"].isin(cohort_stay_ids)].copy()
print(f"Text data after filtering to cohort: {len(df_text)} rows")

# %%
# compute hours from admission to first note
intime_map = df_cohort.set_index("stay_id")["intime"]
df_text["intime"] = pd.to_datetime(df_text["stay_id"].map(intime_map))
df_text["note_time_min"] = pd.to_datetime(df_text["radiology_note_time_min"])
df_text["hours_to_first_note"] = (df_text["note_time_min"] - df_text["intime"]).dt.total_seconds() / 3600

# %%
# keep only notes within 24 hours of admission
text_map = df_text.set_index("stay_id")
within_24h = text_map["hours_to_first_note"].le(24)
valid_notes = text_map.loc[within_24h, "radiology_note_text"]

# build output: one row per cohort stay
out = df_cohort[["stay_id"]].copy()
out["radiology_note"] = out["stay_id"].map(valid_notes)
out["has_radiology_note"] = out["radiology_note"].notna().astype(int)
print(f"With radiology note within 24h: {out['has_radiology_note'].sum()} / {len(out)}")

# %%
# === CXR DETECTION ===

# compile regex patterns for chest x-ray identification
CXR_PATTERNS = [re.compile(p, re.I) for p in [
    r'CHEST\s*(X[\s-]?RAY|RADIOGRAPH)',
    r'CHEST\s*\(.*?(AP|PA|PORT|SINGLE|PRE-OP)',
    r'(AP|PA|PORTABLE)\s+(AP\s+)?(CHEST|VIEW\s+OF\s+THE\s+CHEST)',
    r'EXAMINATION:\s*(CHEST\s*\(|CR\s*-\s*CHEST|CHEST\s+(PORT|SINGLE|PRE-OP|FLUORO))',
    r'TYPE\s+OF\s+EXAMINATION:\s*CHEST',
    r'(SINGLE|PORTABLE)\s+(AP\s+)?(VIEW|RADIOGRAPH)\s+(OF\s+THE\s+)?CHEST',
    r'CHEST\s+PORT',
    r'\bAP\s+CHEST\b',
    r'CHEST\s+\(PA\s+AND\s+LAT',
    r'CHEST\s+\(SINGLE\s+VIEW',
    r'CHEST\s+\(PRE-OP',
    r'\bCHEST\s+RADIOGRAPH\b',
    r'\bCHEST\s+\(PORTABLE',
]]

# split note text into individual reports by --- separator
def split_reports(text):
    if pd.isna(text):
        return []
    return [r.strip() for r in re.split(r'-{3,}', str(text)) if r.strip()]

# check if a report is a CXR based on first 600 chars
def is_cxr_report(report_text):
    header = report_text[:600]
    return any(p.search(header) for p in CXR_PATTERNS)

# %%
# detect CXR for each stay
n = len(out)
out = out.reset_index(drop=True)
is_cxr = np.zeros(n, dtype=int)

for i in range(n):
    note = out.at[i, "radiology_note"]
    for r in split_reports(note):
        if is_cxr_report(r):
            is_cxr[i] = 1
            break

out["isCXRDone"] = is_cxr
print(f"CXR detected: {int(is_cxr.sum())} / {n}")

# %%
# === MEDSPACY NER WITH NEGATION DETECTION ===

import medspacy
from medspacy.ner import TargetRule

nlp = medspacy.load()
target_matcher = nlp.get_pipe("medspacy_target_matcher")

# %%
# define target conditions and their terms
MEDSPACY_TARGETS = {
    "pneumothorax": ["pneumothorax", "pneumothoraces", "ptx"],
    "effusion": ["pleural effusion", "effusions", "effusion", "pleural fluid"],
    "edema": ["edema", "oedema", "pulmonary edema", "pulmonary oedema", "fluid overload"],
    "hemorrhage": [
        "hemorrhage", "haemorrhage", "hemorrhagic", "haemorrhagic",
        "bleeding", "hematoma", "haematoma",
    ],
    "thrombosis": [
        "thrombosis", "thrombus", "thrombi", "thromboembolism", "thrombosed",
        "embolism", "emboli", "dvt", "deep vein thrombosis", "pulmonary embolism",
    ],
    "atelectasis": ["atelectasis", "atelectatic", "atelectases"],
}

# add target rules to the matcher
rules = []
for category, terms in MEDSPACY_TARGETS.items():
    for term in terms:
        rules.append(TargetRule(literal=term, category=category.upper()))
target_matcher.add(rules)

# %%
# flatten all reports for batch processing
all_reports = []
all_indices = []

for i in range(n):
    note = out.at[i, "radiology_note"]
    for report in split_reports(note):
        all_reports.append(report.lower())
        all_indices.append(i)

print(f"Total reports to process: {len(all_reports)}")

# %%
# initialize medspacy columns
for cat in MEDSPACY_TARGETS:
    out[f"medspacy_{cat}"] = 0

# run medspacy NER pipeline
print("Running medspacy NER...")
for doc, idx in zip(nlp.pipe(all_reports, batch_size=256), all_indices):
    for ent in doc.ents:
        cat = ent.label_.lower()
        if cat in MEDSPACY_TARGETS and not ent._.is_negated:
            out.at[idx, f"medspacy_{cat}"] = 1

# print counts per condition
for cat in MEDSPACY_TARGETS:
    col = f"medspacy_{cat}"
    out[col] = out[col].astype(int)
    pos = int(out[col].sum())
    print(f"  {cat}: {pos} ({pos/n*100:.1f}%)")

# %%
# save text features
out.drop(columns=["radiology_note"], inplace=True)
out.to_csv("text_features.csv", index=False)
print(f"Saved text_features.csv: {out.shape}")
print(f"Columns: {list(out.columns)}")
