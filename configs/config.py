"""
configs/config.py
─────────────────
Global constants for the ML4CAD project.
Centralises paths, column names, feature sets, model parameters.
"""

from pathlib import Path
import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"
MODELS_DIR   = PROJECT_ROOT / "models"

# Create directories if they do not exist
for d in [DATA_PROC, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Source files ───────────────────────────────────────────────────────
RAW_DATA_FILE      = DATA_RAW / "raw_data.xlsx"
DATA_PRELIEVO_FILE = DATA_RAW / "data_prelievo.xlsx"
CREATINA_FILE      = DATA_RAW / "creatina_more_columns.xlsx"

# ─── Processed output files ────────────────────────────────────────────
COHORT_STRICT_FILE    = DATA_PROC / "cohort_strict.parquet"
COHORT_COMPETING_FILE = DATA_PROC / "cohort_competing.parquet"
COHORT_SURVIVAL_FILE  = DATA_PROC / "cohort_survival.parquet"
COHORT_FULL_FILE      = DATA_PROC / "cohort_full.parquet"

COHORT_STRICT_CSV    = DATA_PROC / "cohort_strict.csv"
COHORT_COMPETING_CSV = DATA_PROC / "cohort_competing.csv"
COHORT_SURVIVAL_CSV  = DATA_PROC / "cohort_survival.csv"
COHORT_FULL_CSV      = DATA_PROC / "cohort_full.csv"

# Result CSVs
RESULTS_CLF_CSV  = DATA_PROC / "results_clf.csv"
RESULTS_CV_CSV   = DATA_PROC / "results_cv.csv"
RESULTS_TUNE_CSV = DATA_PROC / "results_tune.csv"
RESULTS_SURV_CSV = DATA_PROC / "results_surv.csv"
SHAP_CSV         = DATA_PROC / "shap_importance.csv"
SHAP_PERM_CORR_CSV = DATA_PROC / "shap_permutation_correlation.csv"
DATA_QUALITY_CSV = REPORTS_DIR / "data_quality_corrections.csv"
SURVIVAL_SUMMARY_CSV = REPORTS_DIR / "survival_summary.csv"
RISK_STRATIFICATION_CSV = REPORTS_DIR / "risk_stratification_summary.csv"
COX_COHORT_MULTIVARIATE_CSV = REPORTS_DIR / "cox_cohort_multivariate.csv"
COX_COHORT_SCHOENFELD_CSV = REPORTS_DIR / "cox_cohort_schoenfeld.csv"

# ─── General parameters ────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.30
CV_FOLDS       = 5
HORIZON_DAYS   = 7 * 365.25          # 2556.75 days
HORIZON_YEARS  = 7.0
PRIMARY_CLASSIFICATION_METRIC = "f1_macro"
PRIMARY_CV_METRIC = "f1_macro_opt_mean"

# ─── Column renaming (original names contain newlines) ──────────────────
RENAME = {
    'Gender (Male = 1)':                           'Gender',
    'Age':                                         'Age',
    'Angina':                                      'Angina',
    'Previous CABG':                               'Previous_CABG',
    'Previous PCI':                                'Previous_PCI',
    'Previous Myocardial Infarction':              'Previous_MI',
    'Acute Myocardial Infarction':                 'Acute_MI',
    'Angiography':                                 'Angiography',
    'Vessels':                                     'Vessels',
    'Documented resting \nor exertional ischemia': 'Ischemia',
    'Post-ischemic Dilated\nCardiomyopathy':       'PostIsch_DCM',
    'Smoke\nHistory of smoke':                     'Smoke',
    'Diabetes\nHistory of diabetes':               'Diabetes',
    'Hypertension\nHistory of hypertension':       'Hypertension',
    'Dyslipidemia\nHystory of dyslipidemia':       'Dyslipidemia',
    'Atrial Fibrillation':                         'AFib',
    'fe':                                          'fe',
    'TSH':                                         'TSH',
    'fT3':                                         'fT3',
    'fT4':                                         'fT4',
    'Euthyroid':                                   'Euthyroid',
    'Subclinical primary hypothyroidism (SCH)':    'SCH',
    'Subclinical primary hyperthyroidism\n(SCT)':  'SCT',
    'Low T3':                                      'Low_T3',
    'Ipotiroidismo':                               'Hypothyroid',
    'Ipertiroidismo':                              'Hyperthyroid',
}

# Date columns
DATE_COLS = ['Data prelievo', 'Follow Up Data', 'Data of death']

# Raw outcome columns
OUTCOME_COLS_RAW = ['Total mortality', 'CVD Death',
                    'Fatal MI or Sudden death', 'Cause of death']

# ─── Feature sets ───────────────────────────────────────────────────────
CARDIO_17 = [
    'Gender', 'Age', 'Angina', 'Previous_CABG', 'Previous_PCI',
    'Previous_MI', 'Acute_MI', 'Angiography', 'Vessels', 'Ischemia',
    'PostIsch_DCM', 'Smoke', 'Diabetes', 'Hypertension', 'Dyslipidemia',
    'AFib', 'fe',
]

THYROID_RAW9 = [
    'TSH', 'fT3', 'fT4', 'Euthyroid', 'SCH', 'SCT',
    'Low_T3', 'Hypothyroid', 'Hyperthyroid',
]

# Definition of the 8 feature sets
FEATURE_SETS = {
    'CV17':            CARDIO_17,
    'CV17_THY26':      CARDIO_17 + THYROID_RAW9,
    'CV17_BIN':        CARDIO_17 + ['Thyroid_abnormal'],
    'CV17_ORD':        CARDIO_17 + ['thyroid_ord'],
    'CV17_CONT':       CARDIO_17 + ['TSH', 'fT3', 'fT4'],
    'CV17_CAT':        CARDIO_17 + ['Euthyroid', 'SCH', 'SCT',
                                     'Low_T3', 'Hypothyroid', 'Hyperthyroid'],
    'CV17_RATIO':      CARDIO_17 + THYROID_RAW9 + ['fT3_fT4_ratio'],
    'CV17_RATIO_ONLY': CARDIO_17 + ['TSH', 'fT3', 'fT4', 'fT3_fT4_ratio'],
}

FEATURE_SET_ORDER = list(FEATURE_SETS.keys())

# Continuous features to standardise
CONTINUOUS_FEATURES = ['Age', 'Vessels', 'fe', 'TSH', 'fT3', 'fT4',
                       'fT3_fT4_ratio']

# Ordinal map for thyroid_ord
THYROID_ORD_MAP = {
    'Hypothyroid': -2,
    'SCH':        -1,
    'Euthyroid':   0,
    'Low_T3':      0,   # neutral — not on the hypo-hyper axis
    'SCT':         1,
    'Hyperthyroid': 2,
}

# ─── Models and samplers ───────────────────────────────────────────────
MODEL_NAMES = [
    'LogisticRegression', 'SVC', 'KNeighbors', 'RandomForest',
    'AdaBoost', 'HistGradientBoosting', 'XGBoost', 'MLP',
]

SAMPLER_NAMES = [
    'none', 'RandomUnderSampler', 'SMOTE', 'BorderlineSMOTE', 'SVMSMOTE',
]

# Models for robust CV
ROBUST_CV_MODELS = ['LogisticRegression', 'RandomForest',
                    'HistGradientBoosting', 'XGBoost']
ROBUST_CV_SAMPLER = 'SMOTE'

# Decision-threshold search grid
THRESHOLD_GRID = np.linspace(0.1, 0.9, 41)

# ─── Survival parameters (limited resources: 1 core) ───────────────────
RSF_PARAMS = dict(
    n_estimators=60,
    min_samples_leaf=40,
    max_samples=0.5,
    n_jobs=1,
    random_state=RANDOM_STATE,
)

GBSURV_PARAMS = dict(
    n_estimators=100,
    max_depth=2,
    random_state=RANDOM_STATE,
)
