"""
run_pipeline.py
────────────────
Main pipeline orchestrator for ML4CAD.
Runs all steps in sequence with checkpointing.

Usage:
    python run_pipeline.py              # run entire pipeline
    python run_pipeline.py --force      # recompute everything from scratch
    python run_pipeline.py --step build  # run only one step
    python run_pipeline.py --step eda
    python run_pipeline.py --step screening
    python run_pipeline.py --step robust_cv
    python run_pipeline.py --step tuning
    python run_pipeline.py --step survival
    python run_pipeline.py --step survival_cohort
    python run_pipeline.py --step shap
    python run_pipeline.py --step calibration
"""

import sys, argparse, warnings, time, os
from pathlib import Path

# Force UTF-8 to avoid encoding errors on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                   errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="ML4CAD Pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Recompute everything from scratch")
    parser.add_argument("--step", type=str, default=None,
                        choices=["build", "eda", "screening", "robust_cv",
                                 "tuning", "survival", "survival_cohort",
                                 "shap", "calibration"],
                        help="Run only a specific step")
    args = parser.parse_args()

    force = args.force
    step = args.step
    run_all = step is None

    t_start = time.time()

    print("+" + "=" * 58 + "+")
    print("|" + " ML4CAD - 7-Year Cardiovascular Death Prediction ".center(58) + "|")
    print("+" + "=" * 58 + "+")

    # -- Step 1: Build Dataset --
    if run_all or step == "build":
        from src.preprocessing.build_dataset import run as build_run
        cohorts = build_run(force=force)
    else:
        # Load from file
        import pandas as pd
        from configs.config import (
            COHORT_FULL_FILE, COHORT_STRICT_FILE,
            COHORT_COMPETING_FILE, COHORT_SURVIVAL_FILE,
        )
        cohorts = {
            "full":      pd.read_parquet(COHORT_FULL_FILE),
            "strict":    pd.read_parquet(COHORT_STRICT_FILE),
            "competing": pd.read_parquet(COHORT_COMPETING_FILE),
            "survival":  pd.read_parquet(COHORT_SURVIVAL_FILE),
        }

    # -- Step 2: EDA --
    if run_all or step == "eda":
        from src.visualization.eda import run as eda_run
        eda_run(cohorts["full"], cohorts["strict"], cohorts["competing"])

    # -- Step 3: Classification screening --
    if run_all or step == "screening":
        from src.classification.screening import run as screening_run
        screening_run(
            {"strict": cohorts["strict"], "competing": cohorts["competing"]},
            force=force)

    # -- Step 4: Robust CV --
    if run_all or step == "robust_cv":
        from src.classification.robust_cv import run as cv_run
        cv_run(
            {"strict": cohorts["strict"], "competing": cohorts["competing"]},
            force=force)

    # -- Step 5: Tuning --
    if run_all or step == "tuning":
        from src.classification.tuning import run as tuning_run
        tuning_run(
            {"strict": cohorts["strict"], "competing": cohorts["competing"]},
            force=force)
        from src.classification.tuning import (
            plot_auc_by_feature_set, plot_f1_by_feature_set,
        )
        plot_f1_by_feature_set()
        plot_auc_by_feature_set()

    # -- Step 6: Survival Analysis --
    if run_all or step == "survival":
        from src.survival.survival_analysis import run as surv_run
        surv_run(cohorts["survival"], force=force)

    # -- Step 7: Survival Cohort --
    if run_all or step == "survival_cohort":
        from src.survival.survival_cohort import run as surv_coh_run
        surv_coh_run(
            {"strict": cohorts["strict"], "competing": cohorts["competing"]},
            force=force)

    # -- Step 8: SHAP --
    if run_all or step == "shap":
        from src.evaluation.shap_analysis import run as shap_run
        shap_run(
            {"strict": cohorts["strict"], "competing": cohorts["competing"]},
            force=force)

    # -- Step 9: Calibration --
    if run_all or step == "calibration":
        from src.evaluation.calibration import run as cal_run
        cal_run(
            {"strict": cohorts["strict"], "competing": cohorts["competing"]})

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed/60:.1f} minutes.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
