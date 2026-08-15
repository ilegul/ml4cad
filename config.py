"""Central configuration. This is the only file that normally needs editing."""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Active profile. CHANGE THIS LINE ONLY.
#   smoke   - tiny budget, end-to-end validation of every code path
#   primary - the definitive thesis analysis (expensive, see README)
#   full    - primary plus secondary feature sets and sampling comparison
# ---------------------------------------------------------------------------
PROFILE = "full"

SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "predictions"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "cache"

RAW_CLINICAL = DATA_RAW / "raw_data.xlsx"
RAW_ENROLMENT = DATA_RAW / "data_prelievo.xlsx"
RAW_LABS = DATA_RAW / "creatina_more_columns.xlsx"

COHORT_FULL = DATA_PROC / "cohort_full.parquet"
COHORT_STRICT = DATA_PROC / "cohort_strict_h{horizon}.parquet"
COHORT_SURVIVAL = DATA_PROC / "cohort_survival.parquet"
SPLITS_FILE = DATA_PROC / "splits.json"

# ---------------------------------------------------------------------------
# Raw column mapping. Applied once during preprocessing; every downstream name
# is the canonical English one. Some raw names contain embedded newlines.
# ---------------------------------------------------------------------------
RENAME = {
    "Number": "patient_id",
    "Gender (Male = 1)": "Gender",
    "Age": "Age",
    "Angina": "Angina",
    "Previous CABG": "Previous_CABG",
    "Previous PCI": "Previous_PCI",
    "Previous Myocardial Infarction": "Previous_MI",
    "Acute Myocardial Infarction": "Acute_MI",
    "Angiography": "Angiography",
    "Vessels": "Vessels",
    "Documented resting \nor exertional ischemia": "Ischemia",
    "Post-ischemic Dilated\nCardiomyopathy": "PostIsch_DCM",
    "Smoke\nHistory of smoke": "Smoke",
    "Diabetes\nHistory of diabetes": "Diabetes",
    "Hypertension\nHistory of hypertension": "Hypertension",
    "Dyslipidemia\nHystory of dyslipidemia": "Dyslipidemia",
    "Atrial Fibrillation": "AFib",
    "fe": "LVEF",
    "TSH": "TSH",
    "fT3": "fT3",
    "fT4": "fT4",
    "Euthyroid": "Euthyroid",
    "Subclinical primary hypothyroidism (SCH)": "SCH",
    "Subclinical primary hyperthyroidism\n(SCT)": "SCT",
    "Low T3": "Low_T3",
    "Ipotiroidismo": "Hypothyroid",
    "Ipertiroidismo": "Hyperthyroid",
    "Data prelievo": "enrolment_date",
    "Follow Up Data": "followup_date",
    "Data of death": "death_date",
    "Total mortality": "death_any",
    "CVD Death": "death_cardiac",
    "Cause of death": "cause_of_death",
    "Creatinina": "creatinine",
}

ID_COL = "patient_id"
DATE_COLS = ["enrolment_date", "followup_date", "death_date"]

# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------
# Positive class is the event: y_event = 1 means cardiac death within the
# horizon. p_survive = 1 - p_event is derived only where the paper's
# terminology requires a survival probability.
TARGET = "y_event"
DAYS_PER_YEAR = 365.25

# Explicit patient-level corrections. The raw workbooks are never modified;
# these are applied during preprocessing and logged to
# results/data_corrections.csv so every change stays auditable.
DATA_CORRECTIONS = [
    {
        "patient_id": 6850,
        "issue": "Conflicting thyroid flags: SCH and Hyperthyroid both set",
        "updates": {"SCH": 1, "Hyperthyroid": 0},
        "rationale": "TSH 4.76 is elevated and consistent with subclinical "
                     "hypothyroidism rather than hyperthyroidism.",
    },
    {
        "patient_id": 7286,
        "issue": "Death date present but the all-cause mortality flag is 0",
        "updates": {"death_any": 1},
        "rationale": "A populated death date must be consistent with "
                     "death_any = 1; cardiac death remains 0.",
    },
]

# ---------------------------------------------------------------------------
# Feature sets. Each answers a distinct question; adding more only inflates
# multiplicity, cost and the chance of a spurious result.
# ---------------------------------------------------------------------------
CARDIO = [
    "Gender", "Age", "Angina", "Previous_CABG", "Previous_PCI",
    "Previous_MI", "Acute_MI", "Angiography", "Vessels", "Ischemia",
    "PostIsch_DCM", "Smoke", "Diabetes", "Hypertension", "Dyslipidemia",
    "AFib", "LVEF",
]

# Euthyroid is the reference category and is never included as a dummy.
THYROID_STATES = ["SCH", "SCT", "Low_T3", "Hypothyroid", "Hyperthyroid"]
THYROID_CONT = ["TSH", "fT3", "fT4"]

FEATURE_SETS = {
    "CV17": CARDIO,
    "CV17_THY_CONT": CARDIO + THYROID_CONT,
    "CV17_THY_STATES": CARDIO + THYROID_STATES,
    "CV17_THY_CONT_STATES": CARDIO + THYROID_CONT + THYROID_STATES,
    "CV17_THY_CONT_RATIO": CARDIO + THYROID_CONT + ["fT3_fT4_ratio"],
}

FEATURE_SET_RATIONALE = {
    "CV17": "Cardiovascular predictors alone (baseline).",
    "CV17_THY_CONT": "Do the continuous thyroid biomarkers add value?",
    "CV17_THY_STATES": "Do the clinical thyroid categories add value?",
    "CV17_THY_CONT_STATES": "Do the categories add anything beyond the continuous values?",
    "CV17_THY_CONT_RATIO": "Does the fT3/fT4 ratio add value?",
}

# The prespecified primary contrast. Everything else is secondary.
PRIMARY_BASELINE = "CV17"
PRIMARY_THYROID = "CV17_THY_CONT"
SECONDARY_THYROID = ["CV17_THY_STATES", "CV17_THY_CONT_STATES", "CV17_THY_CONT_RATIO"]

CONTINUOUS_FEATURES = ["Age", "Vessels", "LVEF", "TSH", "fT3", "fT4", "fT3_fT4_ratio"]

# ---------------------------------------------------------------------------
# Models, ensembles, sampling, calibration, thresholds
# ---------------------------------------------------------------------------
MODELS = [
    "LogisticRegression", "SVC", "KNeighbors", "RandomForest",
    "AdaBoost", "MLP", "GradientBoosting", "XGBoost",
]

# Prespecified in Pingitore et al.; always reported, whether or not it wins.
PAPER_ENSEMBLE = ["LogisticRegression", "RandomForest", "AdaBoost"]

# Adapted-ensemble candidates are fixed before validation is touched. The
# top-3 entry is filled from training-only cross-validation ranks.
DIVERSE_ENSEMBLE = ["LogisticRegression", "XGBoost", "MLP"]

# RandomOverSampler, class weighting and SMOTENC respect binary dummies. Plain
# SMOTE interpolates dummies into fractional values, so those variants are
# reported only as a paper-replication sensitivity.
SAMPLERS = ["none", "class_weight", "RandomOverSampler", "SMOTENC"]
SAMPLERS_PAPER_REPLICATION = ["SMOTE", "BorderlineSMOTE", "SVMSMOTE"]

# Sampler used for the development-stage model comparison in notebook 2.
DEV_SAMPLER = "RandomOverSampler"

# Sigmoid is the prespecified primary calibration; isotonic is a sensitivity.
# The calibrator is fitted on training out-of-fold probabilities, never on the
# validation outcomes that later select the threshold.
CALIBRATION_PRIMARY = "sigmoid"
CALIBRATION_SENSITIVITY = "isotonic"
CALIBRATION_BINS = 10
CALIBRATION_CV = 5

# Threshold rule applied to validation probabilities: "f1_macro" or "youden".
THRESHOLD_STRATEGY = "f1_macro"
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)

# Paper stratification cut on p_survive. Kept for comparability only; it is not
# a clinically validated threshold.
PAPER_SURVIVAL_CUT = 0.6
CLINICAL_RISK_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Feature clustering, as in the paper: distance = 1 - abs(Spearman).
N_FEATURE_CLUSTERS = 7

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
PROFILES = {
    "smoke": dict(
        horizons=(7,),
        tuning_feature_sets=("CV17", "CV17_THY_CONT"),
        search_iter=10,
        inner_cv=2,
        outer_cv=3,
        bootstrap=100,
        rsf_search_iter=3,
        rsf_n_estimators=50,
        run_sampling_comparison=True,
        run_paper_replication_samplers=False,
        run_strict_rsf_sensitivity=True,
        run_shap=True,
    ),
    "primary": dict(
        horizons=(7, 10),
        tuning_feature_sets=("CV17", "CV17_THY_CONT"),
        search_iter=5000,
        inner_cv=2,
        outer_cv=5,
        bootstrap=2000,
        rsf_search_iter=20,
        rsf_n_estimators=300,
        run_sampling_comparison=True,
        run_paper_replication_samplers=True,
        run_strict_rsf_sensitivity=True,
        run_shap=True,
    ),
    # Two budgets depart from the paper, both sized from measurement. Search:
    # 1000 draws instead of 5000, because 5000 costs 8.45 h per cell over these
    # spaces and ten cells would take 84 h. This is a computational compromise;
    # run the primary profile to check the primary contrast at 5000. Survival:
    # the paper trains no survival model, so the draw count is sized to keep
    # the extension affordable.
    "full": dict(
        horizons=(7, 10),
        tuning_feature_sets=tuple(FEATURE_SETS),
        search_iter=1000,
        inner_cv=2,
        outer_cv=5,
        bootstrap=2000,
        rsf_search_iter=25,
        rsf_n_estimators=300,
        run_sampling_comparison=True,
        run_paper_replication_samplers=True,
        run_strict_rsf_sensitivity=True,
        run_shap=True,
    ),
}

if PROFILE not in PROFILES:
    raise ValueError(f"Unknown PROFILE {PROFILE!r}; choose from {list(PROFILES)}")

_active = PROFILES[PROFILE]

HORIZONS = _active["horizons"]
TUNING_FEATURE_SETS = _active["tuning_feature_sets"]
SEARCH_ITER = _active["search_iter"]
INNER_CV = _active["inner_cv"]
OUTER_CV = _active["outer_cv"]
BOOTSTRAP = _active["bootstrap"]
RSF_SEARCH_ITER = _active["rsf_search_iter"]
RSF_N_ESTIMATORS = _active["rsf_n_estimators"]
RUN_SAMPLING_COMPARISON = _active["run_sampling_comparison"]
RUN_PAPER_REPLICATION_SAMPLERS = _active["run_paper_replication_samplers"]
RUN_STRICT_RSF_SENSITIVITY = _active["run_strict_rsf_sensitivity"]
RUN_SHAP = _active["run_shap"]

# Split proportions, as in the paper.
TRAIN_FRAC, VALID_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

N_JOBS = 8

# Cached results are reused whenever the signature matches. Set to False to
# force recomputation of every expensive step.
USE_CACHE = True

# Bumping this invalidates every cached artifact of the new pipeline.
PROTOCOL_VERSION = "v1"
