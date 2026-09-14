"""Thesis figures derived from stored results and frozen artifacts.

Every figure here is regenerated from files that the notebooks already
produced (data/processed, predictions, results, models). Nothing is refitted
and no reported number changes. The script writes only into ``figures`` and
``results/thyroid_state_composition.csv``.

    python 8_thesis_figures.py            all figures
    python 8_thesis_figures.py thyroid    one group: cohort, thyroid,
                                          calibration, shap
"""
import importlib
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

import config
import train
import utils

robustness = importlib.import_module("7_robustness")
figure_style, save_png = robustness.figure_style, robustness.save_png

HORIZON = int(config.HORIZONS[0])
ARM = f"h{HORIZON}_{config.PRIMARY_THYROID}_paper"
SET_COLOURS = dict(zip(config.FEATURE_SETS, ["tab:blue", "tab:orange", "tab:green",
                                             "tab:red", "tab:purple"]))
STATES = [("Euthyroid", "Euthyroid (reference)"), ("Low_T3", "Low-T3 syndrome"),
          ("SCH", "Subclinical hypothyroidism"), ("SCT", "Subclinical hyperthyroidism"),
          ("Hypothyroid", "Overt hypothyroidism"), ("Hyperthyroid", "Overt hyperthyroidism")]


# ---------------------------------------------------------------------------
# Cohort flow
# ---------------------------------------------------------------------------

def cohort_numbers():
    raw = pd.read_excel(config.RAW_CLINICAL)
    id_source = [k for k, v in config.RENAME.items() if v == config.ID_COL][0]
    flow = pd.read_csv(config.RESULTS_DIR / "cohort_flow.csv").set_index("horizon")
    survival = utils.read_frame(config.COHORT_SURVIVAL)
    full = utils.read_frame(config.COHORT_FULL)
    return {
        "source_records": len(raw),
        "missing_id": int(raw[id_source].isna().sum()),
        "full": len(full),
        "tte": len(survival),
        "tte_excluded": len(full) - len(survival),
        "tte_cardiac": int(survival["event_cardiac"].sum()),
        "tte_noncardiac": int(survival["event_noncardiac"].sum()),
        "strict": {int(h): dict(n=int(flow.loc[h, "strict_cohort"]),
                                events=int(flow.loc[h, "events_within_horizon"]),
                                censored=int(flow.loc[h, "excluded_censored_before_horizon"]),
                                noncardiac=int(flow.loc[h, "excluded_noncardiac_death_before_horizon"]))
                   for h in flow.index},
    }


def _box(ax, x, y, w, h, title, lines):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.0, edgecolor="0.25", facecolor="white"))
    step = 0.27
    top = y + step * len(lines) / 2
    ax.text(x, top, title, ha="center", va="center", fontsize=9, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x, top - step * (i + 1), line, ha="center", va="center", fontsize=8.5)


def _note(ax, x, y, lines):
    ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=8,
            color="0.25", linespacing=1.3)


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="0.25", linewidth=1.0,
                                shrinkA=0, shrinkB=0))


def figure_cohort_flow():
    n = cohort_numbers()
    assert n["source_records"] - n["missing_id"] == n["full"], "flow does not add up"
    s7, s10 = n["strict"][7], n["strict"][10]
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    bw = 3.25
    xs = (1.75, 5.3, 8.85)
    _box(ax, 5.3, 6.5, bw, 0.9, "Clinical source", [f"{n['source_records']} records"])
    _note(ax, 8.9, 6.5, [f"{n['missing_id']} records excluded:", "patient identifier missing"])
    _arrow(ax, (6.925, 6.5), (7.5, 6.5))
    _box(ax, 5.3, 4.75, bw, 0.9, "Full cohort", [f"{n['full']} patients"])
    _arrow(ax, (5.3, 6.05), (5.3, 5.2))
    y3 = 2.55
    _box(ax, xs[0], y3, bw, 1.3, "Strict cohort, 7 years",
         [f"{s7['n']} patients", f"{s7['events']} cardiac deaths"])
    _box(ax, xs[1], y3, bw, 1.3, "Time-to-event cohort",
         [f"{n['tte']} patients", f"{n['tte_cardiac']} cardiac deaths",
          f"{n['tte_noncardiac']} non-cardiac deaths"])
    _box(ax, xs[2], y3, bw, 1.3, "Strict cohort, 10 years",
         [f"{s10['n']} patients", f"{s10['events']} cardiac deaths"])
    _note(ax, xs[0], 0.95, ["Excluded before the horizon:", f"{s7['censored']} censored alive",
                           f"{s7['noncardiac']} non-cardiac deaths"])
    _note(ax, xs[1], 0.95, [f"{n['tte_excluded']} records excluded:", "recorded follow-up of zero"])
    _note(ax, xs[2], 0.95, ["Excluded before the horizon:", f"{s10['censored']} censored alive",
                           f"{s10['noncardiac']} non-cardiac deaths"])
    ax.plot([5.3, 5.3], [4.3, 3.75], color="0.25", linewidth=1.0)
    ax.plot([xs[0], xs[2]], [3.75, 3.75], color="0.25", linewidth=1.0)
    for x in xs:
        _arrow(ax, (x, 3.75), (x, y3 + 0.65))
    save_png(fig, "cohort_flow")
    plt.close(fig)
    return n


# ---------------------------------------------------------------------------
# Thyroid-state composition
# ---------------------------------------------------------------------------

def thyroid_composition_table():
    cohorts = [("Full cohort", utils.read_frame(config.COHORT_FULL))]
    for h in config.HORIZONS:
        cohorts.append((f"Strict cohort, {int(h)} years",
                        utils.read_frame(str(config.COHORT_STRICT).format(horizon=int(h)))))
    columns = [c for c, _ in STATES]
    rows = []
    for name, frame in cohorts:
        flags = frame[columns]
        if not ((flags.sum(axis=1) == 1).all()):
            raise ValueError(f"thyroid states are not mutually exclusive in {name}")
        for column in columns:
            count = int(frame[column].sum())
            rows.append({"cohort": name, "n": len(frame), "state": column,
                         "count": count, "percent": 100.0 * count / len(frame)})
    table = pd.DataFrame(rows)
    utils.save_result(table, "thyroid_state_composition")
    return table


def figure_thyroid_composition(table):
    cohorts = list(dict.fromkeys(table["cohort"]))
    shades = ["#9ecae1", "#4292c6", "#08306b"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    height = 0.24
    y = np.arange(len(STATES))[::-1]
    for j, cohort in enumerate(cohorts):
        sub = table[table["cohort"] == cohort].set_index("state")
        values = [sub.loc[c, "percent"] for c, _ in STATES]
        pos = y + (1 - j) * height
        n = int(sub["n"].iloc[0])
        ax.barh(pos, values, height=height * 0.92, color=shades[j],
                label=f"{cohort} (n = {n})")
        for p, v in zip(pos, values):
            ax.text(v + 0.6, p, f"{v:.1f}%", va="center", fontsize=8, color="0.2")
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in STATES])
    ax.set_xlabel("percentage of the cohort")
    ax.set_xlim(0, table["percent"].max() * 1.12)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_png(fig, "thyroid_state_composition")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration and decision curves, seven-year test partition
# ---------------------------------------------------------------------------

def figure_calibration_and_decision():
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    for fs in config.TUNING_FEATURE_SETS:
        frame = train.load_test_predictions(f"h{HORIZON}_{fs}_paper")
        points = utils.reliability_points(frame["y_true"], frame["proba_calibrated"])
        ax.plot(points["mean_predicted"], points["observed_rate"], marker="o", markersize=4,
                linewidth=1.3, color=SET_COLOURS[fs], label=fs)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1, label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_png(fig, f"calibration_h{HORIZON}")
    plt.close(fig)

    curves = pd.read_csv(config.RESULTS_DIR / "decision_curves.csv")
    curves = curves[(curves.horizon == HORIZON) & (curves.ensemble == "paper")]
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    for fs in config.TUNING_FEATURE_SETS:
        sub = curves[curves.feature_set == fs]
        ax.plot(sub["threshold"], sub["net_benefit_model"], marker="o", markersize=4,
                linewidth=1.3, color=SET_COLOURS[fs], label=fs)
    reference = curves[curves.feature_set == config.PRIMARY_BASELINE]
    ax.plot(reference["threshold"], reference["net_benefit_treat_all"], linestyle="--",
            color="0.5", linewidth=1, label="treat all")
    ax.axhline(0, linestyle=":", color="0.3", linewidth=1, label="treat none")
    ax.set_xlabel("risk threshold")
    ax.set_ylabel("net benefit")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_png(fig, f"decision_curve_h{HORIZON}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP dependence of the frozen seven-year ensemble
# ---------------------------------------------------------------------------

def shap_values_of_frozen_model():
    import shap

    bundle = train.load_model(f"final_{ARM}")
    predictions = train.load_test_predictions(ARM)
    strict = utils.read_frame(str(config.COHORT_STRICT).format(horizon=HORIZON))
    X, y, ids = utils.extract_xy(strict, bundle["feature_set"])
    masks = utils.split_masks(ids, utils.load_splits(HORIZON))
    X_test = X[masks["test"]]
    utils.assert_same_patients(ids[masks["test"]], predictions[config.ID_COL].to_numpy())

    member = [m for m in bundle["members"]
              if m in ("RandomForest", "GradientBoosting", "XGBoost", "AdaBoost")][0]
    pipe = bundle["members"][member]
    current = X_test.copy()
    for position, (_, step) in enumerate(pipe.steps[:-1]):
        if hasattr(step, "transform"):
            current = step.transform(current if position == 0 else np.asarray(current))
    transformed = pd.DataFrame(np.asarray(current), columns=X_test.columns, index=X_test.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        values = shap.TreeExplainer(pipe.steps[-1][1]).shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    # The stored importance table is the check that the same model is explained.
    stored = pd.read_csv(config.RESULTS_DIR / f"shap_importance_{ARM}.csv").set_index("feature")
    recomputed = pd.Series(np.abs(values).mean(axis=0), index=X_test.columns)
    if not np.allclose(recomputed[stored.index], stored["mean_abs_shap"], atol=1e-8):
        raise ValueError("recomputed SHAP importances differ from the stored table")
    return pd.DataFrame(values, columns=X_test.columns, index=X_test.index), X_test, member


def _dependence_axes(ax, x, s, label):
    ax.scatter(x, s, s=12, alpha=0.5, color="tab:blue", linewidths=0)
    ax.axhline(0, color="0.45", linewidth=0.9, linestyle=":")
    ax.set_xlabel(label)
    ax.set_ylabel(f"SHAP value for {label}")


def figure_shap_dependence():
    values, X_test, member = shap_values_of_frozen_model()
    # Sized for two side-by-side panels at 0.48 of the text width.
    fig, ax = plt.subplots(figsize=(3.7, 3.4))
    _dependence_axes(ax, X_test["Age"], values["Age"], "Age (years)")
    fig.tight_layout()
    save_png(fig, f"shap_dependence_Age_h{HORIZON}")
    plt.close(fig)

    x, s = X_test["fT4"].to_numpy(dtype=float), values["fT4"].to_numpy()
    upper = float(np.percentile(x, 99))
    inside = x <= upper
    fig, ax = plt.subplots(figsize=(3.7, 3.4))
    _dependence_axes(ax, x[inside], s[inside], "fT4 (ng/L)")
    ax.set_title(f"fT4 up to the 99th percentile ({upper:.1f} ng/L)\n"
                 f"{int(inside.sum())} of {len(x)} patients", fontsize=8.5, color="0.25")
    # The lower-right corner holds no observation: every patient above the
    # median fT4 has a positive contribution.
    inset = ax.inset_axes([0.63, 0.06, 0.35, 0.33])
    inset.scatter(x, s, s=6, alpha=0.5, color="tab:blue", linewidths=0)
    inset.axhline(0, color="0.45", linewidth=0.7, linestyle=":")
    inset.add_patch(Rectangle((x.min(), s.min()), upper - x.min(), s.max() - s.min(),
                              fill=False, edgecolor="tab:red", linewidth=0.8))
    inset.set_title("full observed range", fontsize=7.5, color="0.25", pad=2)
    inset.tick_params(labelsize=7)
    inset.grid(False)
    fig.tight_layout()
    save_png(fig, f"shap_dependence_fT4_inset_h{HORIZON}")
    plt.close(fig)
    return member, upper, int(inside.sum()), len(x)


# ---------------------------------------------------------------------------
# Development architecture of the fixed-horizon analysis
# ---------------------------------------------------------------------------

def _plain_box(ax, x, y, w, h, text, face="white", bold=False, size=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.0, edgecolor="0.25", facecolor=face))
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def figure_pipeline():
    """Three partition columns; every box is sized from the measured text.

    Data units are inches, so the figure is laid out at its final size and
    is included in the thesis at the width of the text block without being
    scaled down.
    """
    from matplotlib.font_manager import FontProperties

    size, header_size = 8.0, 8.5
    pad_w, pad_h, line_h, gap, margin = 0.30, 0.22, 0.155, 0.30, 0.12
    fig = plt.figure(figsize=(6.0, 6.0))
    renderer = fig.canvas.get_renderer()

    def text_width(text, pts, bold=False):
        fp = FontProperties(family="DejaVu Sans", size=pts, weight="bold" if bold else "normal")
        return max(renderer.get_text_width_height_descent(line, fp, ismath=False)[0]
                   for line in text.split("\n")) / fig.dpi

    def height(text):
        return (text.count("\n") + 1) * line_h + pad_h

    headers = ["Training partition (60%)", "Validation partition (20%)", "Test partition (20%)"]
    training = ["Random hyperparameter\nsearch (macro-F1,\nstratified two-fold)",
                "Tuned pipelines\n(eight model families)",
                "Training out-of-fold\nprobabilities (five-fold)",
                "Sigmoid calibrator\n(fitted on out-of-fold\nprobabilities)"]
    validation = [None,
                  "Imbalance treatment\nselected per model family",
                  "Adapted ensemble selected\namong three fixed candidates",
                  "Decision threshold\n(macro-F1 over a fixed grid)"]
    frozen = ("Frozen configuration: tuned pipelines,\n"
              "imbalance treatment, ensemble,\n"
              "calibration method and decision threshold")
    evaluation = "Single evaluation of\nthe frozen configuration"
    stored = "Stored predictions for\npaired comparisons and\npost-development analyses"

    col_w = [max([text_width(headers[0], header_size, True)] + [text_width(t, size) for t in training]) + pad_w,
             max([text_width(headers[1], header_size, True)] + [text_width(t, size) for t in validation if t]) + pad_w,
             max([text_width(headers[2], header_size, True), text_width(evaluation, size),
                  text_width(stored, size)]) + pad_w]
    frozen_w = col_w[0] + gap + col_w[1]
    col_w[0] = max(col_w[0], (text_width(frozen, size) + pad_w - gap) / 2)
    col_w[1] = max(col_w[1], (text_width(frozen, size) + pad_w - gap) / 2)
    frozen_w = col_w[0] + gap + col_w[1]
    xt = margin + col_w[0] / 2
    xv = xt + col_w[0] / 2 + gap + col_w[1] / 2
    xs = xv + col_w[1] / 2 + gap + col_w[2] / 2
    total_w = xs + col_w[2] / 2 + margin

    header_h = line_h + pad_h
    row_h = [max(height(t), height(v) if v else 0) for t, v in zip(training, validation)]
    frozen_h, eval_h, stored_h = height(frozen), height(evaluation), height(stored)
    total_h = (margin + header_h + gap + sum(row_h) + gap * len(row_h)
               + max(frozen_h, eval_h) + gap + stored_h + margin)
    fig.set_size_inches(total_w, total_h)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    y = total_h - margin - header_h / 2
    for x, w_, text in zip((xt, xv, xs), col_w, headers):
        _plain_box(ax, x, y, w_, header_h, text, face="0.92", bold=True, size=header_size)
    header_bottom = y - header_h / 2
    rows = []
    y = header_bottom
    for h in row_h:
        y -= gap + h / 2
        rows.append(y)
        y -= h / 2
    for yc, h, t, v in zip(rows, row_h, training, validation):
        _plain_box(ax, xt, yc, col_w[0], h, t, size=size)
        if v:
            _plain_box(ax, xv, yc, col_w[1], h, v, size=size)
    y_frozen = rows[-1] - row_h[-1] / 2 - gap - max(frozen_h, eval_h) / 2
    xf = (xt - col_w[0] / 2 + xv + col_w[1] / 2) / 2
    _plain_box(ax, xf, y_frozen, frozen_w, frozen_h, frozen, size=size)
    _plain_box(ax, xs, y_frozen, col_w[2], eval_h, evaluation, size=size)
    y_stored = y_frozen - max(frozen_h, eval_h) / 2 - gap - stored_h / 2
    _plain_box(ax, xs, y_stored, col_w[2], stored_h, stored, size=size)

    # vertical chains, then the two horizontal hand-overs
    tops = [header_bottom] + [yc - h / 2 for yc, h in zip(rows, row_h)]
    bottoms = [yc + h / 2 for yc, h in zip(rows, row_h)] + [y_frozen + frozen_h / 2]
    for a, b in zip(tops, bottoms):
        _arrow(ax, (xt, a), (xt, b))
    _arrow(ax, (xv, header_bottom), (xv, rows[1] + row_h[1] / 2))
    for a, b in zip(tops[2:], bottoms[2:]):
        _arrow(ax, (xv, a), (xv, b))
    _arrow(ax, (xs, header_bottom), (xs, y_frozen + eval_h / 2))
    _arrow(ax, (xs, y_frozen - eval_h / 2), (xs, y_stored + stored_h / 2))
    _arrow(ax, (xt + col_w[0] / 2, rows[1]), (xv - col_w[1] / 2, rows[1]))
    _arrow(ax, (xf + frozen_w / 2, y_frozen), (xs - col_w[2] / 2, y_frozen))
    save_png(fig, "development_pipeline")
    plt.close(fig)
    return round(total_w, 2), round(total_h, 2)


# ---------------------------------------------------------------------------

def main(argv):
    figure_style(plt)
    groups = set(argv) or {"cohort", "pipeline", "thyroid", "calibration", "shap"}
    if "cohort" in groups:
        print("cohort flow:", figure_cohort_flow())
    if "pipeline" in groups:
        print("development pipeline written, size in inches:", figure_pipeline())
    if "thyroid" in groups:
        table = thyroid_composition_table()
        figure_thyroid_composition(table)
        print(table.pivot(index="state", columns="cohort", values="percent").round(2).to_string())
    if "calibration" in groups:
        figure_calibration_and_decision()
        print("calibration and decision curves written")
    if "shap" in groups:
        print("shap dependence:", figure_shap_dependence())


if __name__ == "__main__":
    main(sys.argv[1:])
