"""Supplementary SHAP, ablation and permutation for frozen classification options.

Run after notebooks 1 to 4 and scripts 9 and 10. Original analyses are read only.
Executed extension outputs and the reselected-C bundles are versioned separately.

Usage:
    python 11_classification_explainability.py --verify
    python 11_classification_explainability.py --summary
    python 11_classification_explainability.py --stage shap
    python 11_classification_explainability.py --stage importance
    python 11_classification_explainability.py --pilot --patients 6
"""

import argparse
import warnings

import pandas as pd
from threadpoolctl import threadpool_limits

import config
from interpretation import classification as shared
from interpretation import importance, shap, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['shap', 'importance', 'all'], default='all')
    parser.add_argument('--horizons', nargs='+', type=int, choices=[7, 10], default=[7, 10])
    parser.add_argument('--feature-sets', nargs='+', choices=list(config.FEATURE_SETS),
                        default=list(config.FEATURE_SETS))
    parser.add_argument('--patients', type=int, default=64)
    parser.add_argument('--background', type=int, default=16)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--pilot', action='store_true', help='Separate SHAP direct/replay validation')
    parser.add_argument('--summary', action='store_true', help='Rebuild reports from completed outputs')
    parser.add_argument('--verify', action='store_true', help='Verify sources and frozen probabilities')
    args = parser.parse_args()
    if min(args.patients, args.background, args.cycles, args.repeats) <= 0:
        parser.error('Sampling and resampling counts must be positive')
    if sum([args.pilot, args.summary, args.verify]) > 1:
        parser.error('Choose one of --pilot, --summary and --verify')
    shared.RESULTS.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        if args.summary:
            from interpretation import report_importance, report_shap
            report_shap.main()
            report_importance.main()
        elif args.pilot:
            shap.run_group(7, 'CV17_THY_CONT', args)
        elif args.verify:
            records = []
            for horizon in args.horizons:
                for feature_set in args.feature_sets:
                    records.extend(verification.verify_group(horizon, feature_set))
            pd.DataFrame(records).to_csv(shared.RESULTS / 'verification.csv', index=False)
        else:
            for horizon in args.horizons:
                for feature_set in args.feature_sets:
                    if args.stage in ('shap', 'all'):
                        shap.run_group(horizon, feature_set, args)
                    if args.stage in ('importance', 'all'):
                        importance.run_group(horizon, feature_set, args.repeats)
    shared.write_inventory()


if __name__ == '__main__':
    main()
