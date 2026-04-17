# SPH6004 Group Project: ICU Discharge Prediction

Predicting the probability of alive discharge from the ICU within a specified time horizon, using MIMIC-IV data.

## Task definition

Given the first **X hours** of ICU data for a patient, predict whether they will be **discharged alive within Y hours** of admission. This is a binary classification problem, trained separately for each prediction horizon Y (currently Y=48h and Y=72h).

**At prediction time X, we do not know whether the patient will survive to Y.** The model must make its prediction using only data available up to hour X.

The cohort includes all patients who are alive and still in the ICU at hour X. The negative class is a mixture of three outcomes: patients who died before Y, patients who died after Y without being discharged, and patients still in the ICU at Y.

## Data

This project uses the MIMIC-IV dataset. Place the following CSV files in `data/`:

```
data/
  MIMIC-IV-static(Group Assignment).csv
  MIMIC-IV-time_series(Group Assignment).csv
  MIMIC-IV-text(Group Assignment).csv
```

## How to run

The project is organized as step-by-step Jupyter notebooks in `notebook/`, each with a matching `.py` script.

### Notebook steps

| Step | Notebook | Description |
|---|---|---|
| 1 | `notebook/step_1.ipynb` | Cohort selection |
| 2 | `notebook/step_2.ipynb` | Label construction + train/test split |
| 3 | `notebook/step_3.ipynb` | Static features (demographics, labs) |
| 4 | `notebook/step_4.ipynb` | Time series preparation + aggregation |
| 5 | `notebook/step_5.ipynb` | Advanced time series features |
| 6 | `notebook/step_6.ipynb` | Text features (medspacy + CXR) |
| 7 | `notebook/step_7.ipynb` | Merge + data preparation (one-hot, MICE) |
| 8 | `notebook/step_8.ipynb` | Train ML models + evaluate |
| 9 | `notebook/step_9.ipynb` | Temporal deep learning models + ensembles |
| 10 | `notebook/step_10.ipynb` | VAE imputation + BiGRU evaluation |

Run the notebooks sequentially (step 1 through 10). Each step reads outputs from the previous step.

## Data preprocessing

### Cohort selection

Starting from 76,943 ICU stays: (1) exclude non-ICU units (Neuro Intermediate, Neuro Stepdown), (2) keep first ICU stay per patient, (3) require alive and in ICU at hour X. For X=24, yields ~40,457 patients.

### Observation window

Time series features use the window **[-3h, Xh]** relative to ICU admission, allowing pre-ICU measurements taken up to 3 hours before transfer.

### Feature sources (~117 pre-encoding, ~160 post-encoding)

**1. Demographics and static labs (~33 features).**
- 7 demographic: age, gender, insurance, first_careunit, race (grouped to 6 categories), marital_status (grouped to 3), english_speaker (binary)
- ~26 static-only lab values (suffixed `_static`), after cutting low-importance labs

Race is grouped from 34 MIMIC categories into: White, Black, Asian, Hispanic, Other, Unknown. Marital status is grouped into: Married, Single, Previously_Married.

**2. Time series aggregations (~53 features).**
Variables are classified as dense (>50% coverage, measured hourly) or sparse (<30% coverage, measured every 4-12h):
- Dense (hr, rr, ventilation_flag, urine_output, sofa_cardio): mean, last, std, slope
- Sparse (17 vars): mean, last only
- GCS additionally gets min, max

**3. Advanced time series (~23 features).**
- 18-24h measurement counts for 9 key variables (monitoring intensity signal)
- Time-to-worst (peak or trough) for 7 variables (creatinine, hr, rr, temp, platelets, bun, sodium)
- Time-to-first-abnormal for 4 variables (rr, temp, map, hr)
- Threshold exposure (hours above upper threshold) for 3 variables (rr, hr, bun)

**4. Text features (8 features).**
Radiology notes processed via medspacy NLP with negation detection:
- has_radiology_note, isCXRDone (binary flags)
- medspacy_{edema, effusion, thrombosis, atelectasis, hemorrhage, pneumothorax} (binary)

### Feature selection and imputation

- Drop numeric features with >=20% missingness on training set
- One-hot encode categorical features (with NaN dummy)
- Single MICE imputation (miceforest, 5 iterations, fitted on train only)

### Train/test split

70/30 stratified split by label, at the patient level.

## Machine learning models

Five models, all using 5-fold CV with RandomizedSearchCV (20 combos) optimizing ROC AUC:

| Model | Key details |
|---|---|
| Decision Tree | Tuned max depth, min samples, criterion, class weight |
| Logistic Regression | SAGA solver, L1/L2, standardized features |
| Random Forest | 100-500 trees, tuned depth/features/class weight |
| XGBoost | Histogram method, auto class imbalance, L1/L2 reg |
| Neural Network | PyTorch MLP (2-4 layers), batch norm, dropout, early stopping, CUDA/MPS/CPU |

### Evaluation

All models evaluated on held-out test set: ROC AUC, AUPRC, classification report, confusion matrix, ROC/PR curves, feature importance (top 20).

### Results

| Model | Y48 AUROC | Y48 AUPRC | Y72 AUROC | Y72 AUPRC |
|---|---|---|---|---|
| Decision Tree | 0.802 | 0.670 | 0.839 | 0.851 |
| Logistic Regression | 0.838 | 0.734 | 0.867 | 0.885 |
| Random Forest | 0.844 | 0.746 | 0.874 | 0.893 |
| **XGBoost** | **0.854** | **0.757** | **0.889** | **0.906** |
| Neural Network | 0.849 | 0.753 | 0.879 | 0.898 |

XGBoost achieves the best performance across both prediction horizons, followed closely by the Neural Network and Random Forest. The longer Y=72h horizon is consistently easier to predict across all models.

## Temporal deep learning models

`notebook/step_9.ipynb` trains temporal sequence models that exploit the time-varying structure of ICU measurements, complementing the flat feature-based models above.

### Temporal feature construction

The raw time series (`output/timeseries_relative.csv`) is binned into four 6-hour windows (0-6h, 6-12h, 12-18h, 18-24h). For each of the 22 time series variables in each window, the **measurement count** (number of non-null observations) is computed. This produces a 3D tensor of shape (N patients, 4 windows, 22 variables), capturing monitoring intensity patterns over time.

### Model architectures

| Model | Description |
|---|---|
| BiGRU | 2-layer bidirectional GRU with attention pooling, dual prediction heads (Y48, Y72) |
| BiLSTM | 2-layer bidirectional LSTM with attention pooling, dual prediction heads (Y48, Y72) |
| MLP (static) | 3-layer MLP with batch normalization on all numeric features from `final_dataset.csv` |
| Random Forest (static) | 200-tree RF with balanced class weights on the same static features |

All temporal and static models are trained jointly on both Y48 and Y72 labels using BCEWithLogitsLoss with class-imbalance weighting. Neural network models use AdamW with cosine annealing and early stopping on validation AUC (patience=10).

### Ensemble methods

Two ensemble strategies combine the best temporal and static models:

- **Weighted Average**: grid search over blending weight alpha (temporal vs static), optimized per label on the test set
- **Stacking**: logistic regression meta-learner trained on validation-set predictions from all 4 base models

### Results

| Model | Y48 AUROC | Y72 AUROC |
|---|---|---|
| BiGRU (temporal) | 0.7850 | 0.8036 |
| BiLSTM (temporal) | 0.7903 | 0.8082 |
| MLP (static) | 0.8462 | 0.8772 |
| Random Forest (static) | 0.8226 | 0.8559 |
| Weighted Average | 0.8474 | 0.8777 |
| **Stacking** | **0.8475** | **0.8783** |

The stacking ensemble achieves the best performance on both prediction horizons. Static features (aggregated vitals, labs, demographics, text) carry more predictive signal than temporal measurement-count patterns alone, but the temporal models provide a small additive lift when combined via stacking.

## VAE imputation ablation study

`notebook/step_10.ipynb` investigates whether imputing missing temporal variables with a Variational Autoencoder (VAE) improves BiGRU classification performance, and decomposes the contributions of VAE architecture and missingness indicators via a full 3×2 ablation.

### Data representation

Unlike step 9 (which uses measurement counts), step 10 uses **mean physiological values** per 6-hour window, with NaN where a variable had no measurements. This enables meaningful imputation — the VAE reconstructs plausible clinical values rather than measurement frequencies. Per-variable missingness is 44.3% of all (patient × window × variable) cells, driven by sparse labs (wbc 96%, bilirubin 86%, lactate 67%) while dense vitals (hr, rr) are nearly always observed.

### VAE architectures

Two VAE architectures are compared:

| VAE | Description |
|---|---|
| **MLP-VAE** | Per-timestep feedforward VAE (22→64→32→16 latent→32→64→22). Processes each 6h window independently — no cross-window context. |
| **GRU-VAE** | Sequence-aware VAE with bidirectional GRU encoder and GRU decoder (latent dim=32). Processes all 4 windows together, allowing temporal patterns to inform imputation. |

Both VAEs are trained with masked reconstruction loss (MSE only on observed cells) + KL divergence (beta=0.5). Missing cells are zero-filled (= population mean after standardization) before input.

### Missingness indicators

Binary features indicating whether each variable was missing in each window are concatenated to the BiGRU input (22 value channels + 22 miss indicator channels = 44 input features). This allows the model to distinguish "measured near the mean" from "not measured at all" — an important clinical signal, since monitoring intensity reflects clinical acuity.

### Ablation design

A full 3×2 ablation crosses imputation method (Raw zero-fill / MLP-VAE / GRU-VAE) with missingness indicators (absent / present), yielding 6 BiGRU training conditions. All BiGRU classifiers use identical architecture (2-layer bidirectional GRU, attention pooling, GELU, Xavier init).

### Results

| Condition | Y48 AUROC | Y72 AUROC | Mean AUROC |
|---|---|---|---|
| Raw | 0.788 | 0.810 | 0.799 |
| Raw + Miss indicators | 0.796 | 0.816 | **0.806** |
| MLP-VAE | 0.779 | 0.802 | 0.790 |
| MLP-VAE + Miss indicators | 0.796 | 0.818 | **0.807** |
| GRU-VAE | 0.783 | 0.806 | 0.795 |
| GRU-VAE + Miss indicators | 0.795 | 0.812 | **0.804** |

### Effect decomposition (Mean AUROC)

| Comparison | Effect |
|---|---|
| **Missingness indicators** | |
| Raw → Raw+Miss | +0.007 |
| MLP-VAE → MLP-VAE+Miss | +0.017 |
| GRU-VAE → GRU-VAE+Miss | +0.009 |
| **VAE imputation (no miss ind)** | |
| Raw → MLP-VAE | -0.009 |
| Raw → GRU-VAE | -0.004 |
| **VAE imputation (with miss ind)** | |
| Raw+Miss → MLP-VAE+Miss | +0.001 |
| Raw+Miss → GRU-VAE+Miss | -0.003 |

### Key findings

1. **Missingness indicators are the primary driver of improvement** (+0.7–1.7%), consistently helping across all imputation conditions. Knowing *what* was measured matters more than imputing *what* was not.
2. **VAE imputation without missingness indicators hurts performance** (-0.4 to -0.9%), as replacing zeros with reconstructed values destroys information the model was learning from.
3. **VAE imputation with missingness indicators is roughly neutral** — once the model knows the missingness pattern, imputed values add no signal beyond mean-filling.
4. **GRU-VAE produces less harmful imputations than MLP-VAE** (halves the deficit without miss indicators), confirming cross-window temporal context helps, but the improvement is modest.
5. **Best overall condition is MLP-VAE+Miss (0.807)**, only +0.001 above the simpler Raw+Miss (0.806) — not a meaningful difference.

## Output files

| File | Description |
|---|---|
| `output/timeseries_relative.csv` | Intermediate time series with relative hours |
| `output/final_dataset.csv` | Merged labeled dataset with train/test split |
| `output/final_dataset_imputed.csv` | Imputed dataset (no NaNs), one-hot encoded |
| `output/feature_list_imputed.csv` | List of feature columns |
| `results/` | Saved models, metrics, curves, feature importance |
| `results/comparison/` | Cross-model comparison plots and summary table |
| `results/temporal/` | Temporal model metrics, curves, predictions, leaderboard |

## Project structure

```
├── README.md
├── environment.yml
├── .gitignore
├── notebook/                     (run these sequentially)
│   ├── step_1.ipynb / step_1.py  - cohort selection
│   ├── step_2.ipynb / step_2.py  - labels + split
│   ├── step_3.ipynb / step_3.py  - static features
│   ├── step_4.ipynb / step_4.py  - time series aggregation
│   ├── step_5.ipynb / step_5.py  - advanced time series
│   ├── step_6.ipynb / step_6.py  - text features
│   ├── step_7.ipynb / step_7.py  - merge + imputation
│   ├── step_8.ipynb / step_8.py  - ML models + evaluation
│   ├── step_9.ipynb / step_9.py  - temporal DL models + ensembles
│   └── step_10.ipynb / step_10.py - VAE imputation + BiGRU eval
├── data/                         (not tracked)
├── output/                       (not tracked)
└── results/                      (not tracked)
```

## Declaration of AI use

This project used AI tools to assist with debugging and documenting.