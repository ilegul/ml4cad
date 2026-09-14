"""Post hoc robustness analyses of the frozen primary and survival results.

This step was added after the main analysis was complete and does not alter
it. It reads the frozen artifacts (stored split, tuning manifests, sampler
choices, frozen test predictions) and writes only into
``results/robustness`` and ``figures``. Nothing here is used to select a
model or a configuration; every analysis is reported regardless of outcome.

Phases, in execution order:

  D3_k2_check  reproduction gate: the notebook-3 flow re-implemented here
               must reproduce the frozen test predictions and intervals
  A            the 60/20/20 procedure repeated over 50 random partitions
               with hyperparameters and samplers frozen at the reported
               values (10 x 5-fold; a stratified 25 % of each training
               fold is held out for the threshold, so proportions match)
  B1, B2, B5   analyses B (Cox, RSF at the default configuration) and C
               with ten different outer-fold partitions
  B3, B4       bootstrap with model refitting: fit on the bootstrap sample,
               paired concordance difference on the out-of-bag patients
  D1, D2       hyperparameter search of the reference-ensemble members with
               3 and 5 inner folds (same candidates), compared with the
               two-fold winners on a training-only yardstick
  D3_k3, D3_k5 the frozen-test result under the 3- and 5-fold winners; a
               second reading of the test partition, reported in full and
               never used for selection

Usage:
    python 7_robustness.py                 run every phase, then summary,
                                           tables and figures
    python 7_robustness.py A B1 --summary  run selected phases and summarise
    python 7_robustness.py --smoke         reduced sizes, separate output dir
    python 7_robustness.py --summary --tables --figures
                                           rebuild the derived outputs only

Every phase is resumable: a phase whose output exists is skipped.
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sksurv.metrics import concordance_index_censored

import config
import ensemble as ens
import survival as surv
import train
import utils

SEED = config.SEED
BASE, THY = config.PRIMARY_BASELINE, config.PRIMARY_THYROID
MEMBERS = list(config.PAPER_ENSEMBLE)

# Sizes; overridden by smoke mode.
S = dict(
    out=config.RESULTS_DIR / "robustness",
    horizons=tuple(int(h) for h in config.HORIZONS),
    a_reps=10, a_boot=500, a_workers=10,
    b_seeds=[SEED] + list(range(1, 10)), b_boot=1000, b_workers=5,
    oob_boot=2000, oob_workers=8,
    d_iter=1000, d_ks=(3, 5), yard_reps=5, yard_k=3, down_boot=config.BOOTSTRAP,
    rsf_trees=config.RSF_N_ESTIMATORS,
)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(S["out"] / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Frozen configuration of the reported run (read-only)
# ---------------------------------------------------------------------------

def frozen_params(horizon, feature_set, model):
    path = config.MODELS_DIR / f"tuned_h{horizon}_{feature_set}_{model}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


def frozen_sampler(horizon, feature_set, model):
    table = pd.read_csv(config.RESULTS_DIR / "sampler_choice.csv")
    row = table[(table["horizon"] == horizon) & (table["feature_set"] == feature_set)
                & (table["model"] == model)]
    return str(row["sampler"].iloc[0])


def strict_data(horizon, feature_set):
    strict = utils.read_frame(str(config.COHORT_STRICT).format(horizon=horizon))
    X, y, ids = utils.extract_xy(strict, feature_set)
    return X, y, ids


def split_indices(horizon, ids):
    masks = utils.split_masks(ids, utils.load_splits(horizon))
    return {k: np.where(v)[0] for k, v in masks.items()}


# ---------------------------------------------------------------------------
# One arm: members -> ensemble -> sigmoid calibration on training OOF
# ---------------------------------------------------------------------------

def fit_arm(X, y, idx_train, idx_valid, idx_test, feature_set, params, samplers,
            seed=SEED, cal_cv=config.CALIBRATION_CV):
    Xtr, ytr = X.iloc[idx_train], y[idx_train]
    oofs, valids, tests = [], [], []
    for m in MEMBERS:
        pipe = train.make_pipeline(m, samplers[m], feature_set, seed)
        pipe.set_params(**params[m])
        folds = StratifiedKFold(n_splits=cal_cv, shuffle=True, random_state=seed)
        oofs.append(cross_val_predict(clone(pipe), Xtr, ytr, cv=folds,
                                      method="predict_proba", n_jobs=1)[:, 1])
        pipe.fit(Xtr, ytr)
        valids.append(pipe.predict_proba(X.iloc[idx_valid])[:, 1])
        tests.append(pipe.predict_proba(X.iloc[idx_test])[:, 1])
    oof, pv, pt = ens.mean_proba(oofs), ens.mean_proba(valids), ens.mean_proba(tests)
    cal = train.fit_calibrator(oof, ytr, config.CALIBRATION_PRIMARY)
    return {"valid_raw": pv, "test_raw": pt,
            "valid_cal": train.apply_calibrator(cal, pv),
            "test_cal": train.apply_calibrator(cal, pt)}


def three_arms(y, idx_train, idx_valid, idx_test, Xb, Xt,
               base_params, base_samplers, thy_params, thy_samplers):
    base = fit_arm(Xb, y, idx_train, idx_valid, idx_test, BASE, base_params, base_samplers)
    thr_base = utils.optimize_threshold(y[idx_valid], base["valid_cal"])
    locked = fit_arm(Xt, y, idx_train, idx_valid, idx_test, THY, base_params, base_samplers)
    own = fit_arm(Xt, y, idx_train, idx_valid, idx_test, THY, thy_params, thy_samplers)
    thr_own = utils.optimize_threshold(y[idx_valid], own["valid_cal"])
    return base, thr_base, locked, own, thr_own


def arm_rows(yt, base, thr_base, locked, own, thr_own, n_boot, tag):
    metrics = []
    for arm_name, arm, thr in [("base", base, thr_base), ("thy_locked", locked, thr_base),
                               ("thy_own", own, thr_own)]:
        for label, col in [("calibrated", "test_cal"), ("uncalibrated", "test_raw")]:
            metrics.append({**tag, "arm": arm_name, "probabilities": label,
                            **utils.classification_metrics(yt, arm[col], thr)})
    deltas = []
    for comparison, other, thr_other in [("locked pipeline", locked, thr_base),
                                         ("independently optimized", own, thr_own)]:
        d = utils.paired_delta(yt, base["test_cal"], other["test_cal"], thr_base,
                               thr_other, n_boot=n_boot, seed=SEED)
        d.insert(0, "comparison", comparison)
        for k, v in reversed(list(tag.items())):
            d.insert(0, k, v)
        d["threshold_base"] = thr_base
        d["threshold_other"] = thr_other
        deltas.append(d)
    return metrics, pd.concat(deltas, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase A: the reported 60/20/20 procedure repeated over random partitions,
# hyperparameters and samplers frozen at the reported values
# ---------------------------------------------------------------------------

def _phase_a_partition(horizon, rep, fold, idx_tr80, idx_test, n_boot):
    Xb, y, _ = strict_data(horizon, BASE)
    Xt, _, _ = strict_data(horizon, THY)
    idx_train, idx_valid = train_test_split(
        idx_tr80, test_size=0.25, stratify=y[idx_tr80],
        random_state=SEED + 1000 * rep + fold)
    base_params = {m: frozen_params(horizon, BASE, m) for m in MEMBERS}
    base_samplers = {m: frozen_sampler(horizon, BASE, m) for m in MEMBERS}
    thy_params = {m: frozen_params(horizon, THY, m) for m in MEMBERS}
    thy_samplers = {m: frozen_sampler(horizon, THY, m) for m in MEMBERS}
    started = time.perf_counter()
    base, thr_base, locked, own, thr_own = three_arms(
        y, idx_train, idx_valid, idx_test, Xb, Xt,
        base_params, base_samplers, thy_params, thy_samplers)
    tag = {"horizon": horizon, "rep": rep, "fold": fold,
           "n_train": len(idx_train), "n_valid": len(idx_valid), "n_test": len(idx_test)}
    metrics, deltas = arm_rows(y[idx_test], base, thr_base, locked, own, thr_own, n_boot, tag)
    for row in metrics:
        row["elapsed_s"] = round(time.perf_counter() - started, 1)
    return metrics, deltas


def phase_a():
    out_m = S["out"] / "A_partitions_metrics.csv"
    out_d = S["out"] / "A_partitions_deltas.csv"
    if out_d.exists():
        log("A: already done")
        return
    all_m, all_d = [], []
    for horizon in S["horizons"]:
        _, y, _ = strict_data(horizon, BASE)
        jobs = []
        for rep in range(S["a_reps"]):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
            for fold, (tr80, te) in enumerate(skf.split(np.zeros(len(y)), y)):
                jobs.append((horizon, rep, fold, tr80, te))
        log(f"A: horizon {horizon}, {len(jobs)} partitions, {S['a_workers']} workers")
        results = Parallel(n_jobs=S["a_workers"], backend="loky")(
            delayed(_phase_a_partition)(h, r, f, tr, te, S["a_boot"])
            for h, r, f, tr, te in jobs)
        for metrics, deltas in results:
            all_m.extend(metrics)
            all_d.append(deltas)
        log(f"A: horizon {horizon} done")
    pd.DataFrame(all_m).to_csv(out_m, index=False)
    pd.concat(all_d, ignore_index=True).to_csv(out_d, index=False)


# ---------------------------------------------------------------------------
# Phase B: survival, dependence on the outer-fold partition and refit bootstrap
# ---------------------------------------------------------------------------

def _survival_cohort():
    return surv.survival_frame(utils.read_frame(config.COHORT_SURVIVAL), None)


def _cindex_frames(cohort, base_risk, other_risk):
    base = pd.DataFrame({config.ID_COL: cohort[config.ID_COL].to_numpy(),
                         "surv_time": cohort["surv_time"].to_numpy(dtype=float),
                         "surv_event": cohort["surv_event"].to_numpy(dtype=int),
                         "proba_calibrated": base_risk})
    other = base.copy()
    other["proba_calibrated"] = other_risk
    return base, other


def _phase_b_seed(model_name, seed, n_boot, rsf_trees):
    config.RSF_N_ESTIMATORS = rsf_trees
    cohort = _survival_cohort()
    folds = utils.make_folds(cohort["surv_event"].to_numpy(), n_splits=5, seed=seed)
    risks, fold_rows = {}, []
    started = time.perf_counter()
    for fs in (BASE, THY):
        X = cohort[utils.feature_list(fs)].copy()
        res = surv.evaluate_survival_cv(X, cohort, model_name, folds,
                                        horizons=config.HORIZONS, n_iter=0)
        risks[fs] = res["oof_risk"]
        block = res["folds"]
        block.insert(0, "feature_set", fs)
        block.insert(0, "seed", seed)
        fold_rows.append(block)
    base, other = _cindex_frames(cohort, risks[BASE], risks[THY])
    delta = surv.paired_cindex_delta(base, other, n_boot=n_boot, seed=SEED)
    folds_table = pd.concat(fold_rows, ignore_index=True)
    row = {"model": model_name, "seed": seed,
           "c_harrell_base": folds_table[folds_table.feature_set == BASE]["c_harrell"].mean(),
           "c_harrell_thy": folds_table[folds_table.feature_set == THY]["c_harrell"].mean(),
           **delta, "elapsed_s": round(time.perf_counter() - started, 1)}
    return row, folds_table


def phase_b_cox():
    out = S["out"] / "B1_cox_partitions.csv"
    if out.exists():
        log("B1: already done")
        return
    log(f"B1: Cox, {len(S['b_seeds'])} fold partitions")
    results = Parallel(n_jobs=S["b_workers"], backend="loky")(
        delayed(_phase_b_seed)("Cox", s, S["b_boot"], S["rsf_trees"]) for s in S["b_seeds"])
    pd.concat([f for _, f in results], ignore_index=True).to_csv(
        S["out"] / "B1_cox_partitions_folds.csv", index=False)
    pd.DataFrame([r for r, _ in results]).to_csv(out, index=False)
    log("B1: done")


def phase_b_rsf():
    out = S["out"] / "B5_rsf_default_partitions.csv"
    if out.exists():
        log("B5: already done")
        return
    rows, folds = [], []
    for seed in S["b_seeds"]:
        log(f"B5: RSF default configuration, fold partition seed {seed}")
        row, table = _phase_b_seed("RSF", seed, S["b_boot"], S["rsf_trees"])
        rows.append(row)
        folds.append(table)
    pd.concat(folds, ignore_index=True).to_csv(
        S["out"] / "B5_rsf_default_partitions_folds.csv", index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    log("B5: done")


def _phase_b_cif_seed(seed, n_boot):
    frame = surv.competing_frame(utils.read_frame(config.COHORT_SURVIVAL))
    horizons = [float(h) for h in config.HORIZONS]
    folds = utils.make_folds(frame["cause_code"].to_numpy(), 5, seed=seed)
    started = time.perf_counter()
    res = {fs: surv.evaluate_cif_cv(frame[utils.feature_list(fs)].copy(), frame, folds, horizons)
           for fs in (BASE, THY)}
    rows = []
    for h in horizons:
        truncated = surv.survival_frame(frame, h)[[config.ID_COL, "surv_time", "surv_event"]]
        base = truncated.assign(cif=res[BASE]["oof_cif"][h])
        other = truncated.assign(cif=res[THY]["oof_cif"][h])
        delta = surv.paired_cindex_delta(base, other, score_col="cif", n_boot=n_boot, seed=SEED)
        fb, ft = res[BASE]["folds"], res[THY]["folds"]
        rows.append({"seed": seed, "horizon": h,
                     "c_harrell_base": fb[fb.horizon == h]["c_harrell"].mean(),
                     "c_harrell_thy": ft[ft.horizon == h]["c_harrell"].mean(),
                     **delta, "elapsed_s": round(time.perf_counter() - started, 1)})
    return rows


def phase_b_cif():
    out = S["out"] / "B2_cif_partitions.csv"
    if out.exists():
        log("B2: already done")
        return
    log(f"B2: analysis C, {len(S['b_seeds'])} fold partitions")
    results = Parallel(n_jobs=S["b_workers"], backend="loky")(
        delayed(_phase_b_cif_seed)(s, S["b_boot"]) for s in S["b_seeds"])
    pd.DataFrame([r for rows in results for r in rows]).to_csv(out, index=False)
    log("B2: done")


def _oob_chunk(kind, draws):
    raw = utils.read_frame(config.COHORT_SURVIVAL)
    frame = surv.survival_frame(raw, None) if kind == "coxB" else surv.competing_frame(raw)
    y = surv.make_surv_y(frame)
    Xs = {fs: frame[utils.feature_list(fs)].copy() for fs in (BASE, THY)}
    n = len(frame)
    horizons = [float(h) for h in config.HORIZONS]
    rows = []
    for draw in draws:
        rng = np.random.default_rng(SEED * 100_003 + int(draw))
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        try:
            if kind == "coxB":
                c = {}
                for fs in (BASE, THY):
                    pipe = surv.survival_pipeline("Cox").fit(Xs[fs].iloc[idx], y[idx])
                    c[fs] = concordance_index_censored(
                        y[oob]["event"], y[oob]["time"], pipe.predict(Xs[fs].iloc[oob]))[0]
                rows.append({"draw": int(draw), "horizon": np.nan, "n_oob": len(oob),
                             "c_base": c[BASE], "c_thy": c[THY], "delta": c[THY] - c[BASE]})
            else:
                cifs = {}
                for fs in (BASE, THY):
                    models = surv.fit_cause_specific(Xs[fs].iloc[idx], frame.iloc[idx])
                    cifs[fs] = surv.predict_cardiac_cif(models, Xs[fs].iloc[oob], horizons)
                for j, h in enumerate(horizons):
                    ev = y[oob]["event"] & (y[oob]["time"] <= h)
                    tm = np.minimum(y[oob]["time"], h)
                    c = {fs: concordance_index_censored(ev, tm, cifs[fs][:, j])[0] for fs in cifs}
                    rows.append({"draw": int(draw), "horizon": h, "n_oob": len(oob),
                                 "c_base": c[BASE], "c_thy": c[THY], "delta": c[THY] - c[BASE]})
        except Exception as error:
            rows.append({"draw": int(draw), "horizon": np.nan, "n_oob": len(oob),
                         "c_base": np.nan, "c_thy": np.nan, "delta": np.nan,
                         "error": type(error).__name__})
    return rows


def phase_b_oob(kind):
    name = {"coxB": "B3_oob_refit_cox", "cifC": "B4_oob_refit_cif"}[kind]
    out = S["out"] / f"{name}.csv"
    if out.exists():
        log(f"{name}: already done")
        return
    chunks = [c for c in np.array_split(np.arange(S["oob_boot"]), S["oob_workers"] * 4) if len(c)]
    log(f"{name}: {S['oob_boot']} out-of-bag refit draws, {len(chunks)} chunks")
    results = Parallel(n_jobs=S["oob_workers"], backend="loky")(
        delayed(_oob_chunk)(kind, list(c)) for c in chunks)
    pd.DataFrame([r for rows in results for r in rows]).to_csv(out, index=False)
    log(f"{name}: done")


# ---------------------------------------------------------------------------
# Phase D: the inner-fold count of the hyperparameter search
# ---------------------------------------------------------------------------

def _tune_key(h, fs, m):
    return f"h{h}_{fs}_{m}"


def phase_d_tune(k):
    out_csv = S["out"] / f"D1_tuning_k{k}.csv"
    out_json = S["out"] / f"D1_params_k{k}.json"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    params = json.loads(out_json.read_text()) if out_json.exists() else {}
    for h in S["horizons"]:
        for fs in (BASE, THY):
            X, y, ids = strict_data(h, fs)
            idx = split_indices(h, ids)
            for m in MEMBERS:
                key = _tune_key(h, fs, m)
                if key in params:
                    continue
                log(f"D1: k={k} tuning {key} ({S['d_iter']} draws)")
                res = train.tune(X.iloc[idx["train"]], y[idx["train"]], m,
                                 config.DEV_SAMPLER, fs, n_iter=S["d_iter"], cv=k)
                res.pop("estimator")
                params[key] = res.pop("best_params")
                rows.append({"k": k, "horizon": h, **res})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                out_json.write_text(json.dumps(params, indent=1))
    log(f"D1: k={k} done")


def params_for(k):
    if k == 2:
        return lambda h, fs, m: frozen_params(h, fs, m)
    table = json.loads((S["out"] / f"D1_params_k{k}.json").read_text())
    return lambda h, fs, m: table[_tune_key(h, fs, m)]


def _yardstick_fold(h, fs, rep_fold, tr, te, ks):
    X, y, _ = strict_data(h, fs)
    rows = []
    for k in ks:
        pf = params_for(k)
        probas = {}
        for m in MEMBERS:
            pipe = train.make_pipeline(m, config.DEV_SAMPLER, fs).set_params(**pf(h, fs, m))
            pipe.fit(X.iloc[tr], y[tr])
            probas[m] = pipe.predict_proba(X.iloc[te])[:, 1]
            rows.append({"horizon": h, "feature_set": fs, "k": k, "rep_fold": rep_fold,
                         "model": m, **utils.classification_metrics(y[te], probas[m], 0.5)})
        rows.append({"horizon": h, "feature_set": fs, "k": k, "rep_fold": rep_fold,
                     "model": "paper_ensemble",
                     **utils.classification_metrics(
                         y[te], ens.mean_proba(list(probas.values())), 0.5)})
    return rows


def phase_d_yardstick():
    out = S["out"] / "D2_training_yardstick.csv"
    if out.exists():
        log("D2: already done")
        return
    ks = [2] + [k for k in S["d_ks"] if (S["out"] / f"D1_params_k{k}.json").exists()]
    jobs = []
    for h in S["horizons"]:
        for fs in (BASE, THY):
            _, y, ids = strict_data(h, fs)
            tr_idx = split_indices(h, ids)["train"]
            rskf = RepeatedStratifiedKFold(n_splits=S["yard_k"], n_repeats=S["yard_reps"],
                                           random_state=SEED)
            for i, (a, b) in enumerate(rskf.split(np.zeros(len(tr_idx)), y[tr_idx])):
                jobs.append((h, fs, i, tr_idx[a], tr_idx[b]))
    log(f"D2: training-only yardstick, ks={ks}, {len(jobs)} folds")
    results = Parallel(n_jobs=S["a_workers"], backend="loky")(
        delayed(_yardstick_fold)(h, fs, i, tr, te, ks) for h, fs, i, tr, te in jobs)
    pd.DataFrame([r for rows in results for r in rows]).to_csv(out, index=False)
    log("D2: done")


def downstream_primary(horizon, pf, n_boot):
    """Notebook-3 flow for the paper ensemble on the primary contrast."""
    data = {fs: strict_data(horizon, fs) for fs in (BASE, THY)}
    Xb, y, ids = data[BASE]
    Xt = data[THY][0]
    idx = split_indices(horizon, ids)
    samplers, sampler_rows = {}, []
    for fs in (BASE, THY):
        Xf = data[fs][0]
        for m in MEMBERS:
            cand = []
            for s in config.SAMPLERS:
                if s == "class_weight" and not train.supports_class_weight(m):
                    continue
                pipe = train.make_pipeline(m, s, fs).set_params(**pf(horizon, fs, m))
                pipe.fit(Xf.iloc[idx["train"]], y[idx["train"]])
                pv = pipe.predict_proba(Xf.iloc[idx["valid"]])[:, 1]
                cand.append({"horizon": horizon, "feature_set": fs, "model": m, "sampler": s,
                             **utils.classification_metrics(y[idx["valid"]], pv, 0.5)})
            ranked = pd.DataFrame(cand).sort_values(["f1_macro", "roc_auc", "sampler"],
                                                    ascending=[False, False, True])
            samplers[(fs, m)] = str(ranked.iloc[0]["sampler"])
            sampler_rows.extend(cand)
    base_params = {m: pf(horizon, BASE, m) for m in MEMBERS}
    thy_params = {m: pf(horizon, THY, m) for m in MEMBERS}
    base_s = {m: samplers[(BASE, m)] for m in MEMBERS}
    thy_s = {m: samplers[(THY, m)] for m in MEMBERS}
    base, thr_base, locked, own, thr_own = three_arms(
        y, idx["train"], idx["valid"], idx["test"], Xb, Xt,
        base_params, base_s, thy_params, thy_s)
    tag = {"horizon": horizon}
    metrics, deltas = arm_rows(y[idx["test"]], base, thr_base, locked, own, thr_own, n_boot, tag)
    chosen = pd.DataFrame([{"horizon": horizon, "feature_set": fs, "model": m, "sampler": s}
                           for (fs, m), s in samplers.items()])
    preds = pd.DataFrame({config.ID_COL: ids[idx["test"]], "y_true": y[idx["test"]],
                          "base_cal": base["test_cal"], "locked_cal": locked["test_cal"],
                          "own_cal": own["test_cal"]})
    return metrics, deltas, chosen, pd.DataFrame(sampler_rows), preds


def phase_d_downstream(k):
    out = S["out"] / f"D3_downstream_k{k}_deltas.csv"
    if out.exists():
        log(f"D3: k={k} already done")
        return
    pf = params_for(k)
    all_m, all_d, all_c, all_s, all_p = [], [], [], [], {}
    for h in S["horizons"]:
        log(f"D3: k={k} downstream horizon {h}")
        m, d, c, s, p = downstream_primary(h, pf, S["down_boot"])
        all_m.extend(m)
        all_d.append(d)
        all_c.append(c)
        all_s.append(s)
        all_p[h] = p
    pd.DataFrame(all_m).to_csv(S["out"] / f"D3_downstream_k{k}_metrics.csv", index=False)
    pd.concat(all_c).to_csv(S["out"] / f"D3_downstream_k{k}_samplers.csv", index=False)
    pd.concat(all_s).to_csv(S["out"] / f"D3_downstream_k{k}_sampler_comparison.csv", index=False)
    for h, p in all_p.items():
        p.to_csv(S["out"] / f"D3_downstream_k{k}_h{h}_test_predictions.csv", index=False)
    pd.concat(all_d, ignore_index=True).to_csv(out, index=False)
    if k == 2:
        _reproduction_check(all_p)
    log(f"D3: k={k} done")


def _reproduction_check(preds_by_h):
    """The k=2 downstream must reproduce the frozen artifacts exactly."""
    rows = []
    for h, p in preds_by_h.items():
        for col, name in [("base_cal", f"h{h}_{BASE}_paper"), ("own_cal", f"h{h}_{THY}_paper"),
                          ("locked_cal", f"h{h}_{THY}_paper_locked")]:
            frozen = pd.read_csv(config.PRED_DIR / f"{name}_test.csv")
            merged = p.merge(frozen, on=config.ID_COL, validate="one_to_one")
            rows.append({"horizon": h, "artifact": name, "n": len(merged),
                         "max_abs_diff": float((merged[col] - merged["proba_calibrated"]).abs().max()),
                         "mismatches": int((merged["y_true_x"] != merged["y_true_y"]).sum())})
    mine = pd.read_csv(S["out"] / "D3_downstream_k2_deltas.csv")
    ref = pd.read_csv(config.RESULTS_DIR / "incremental_value.csv")
    ref = ref[ref.comparison_role == "primary"]
    merged = mine.merge(ref, on=["horizon", "comparison", "metric"],
                        suffixes=("_recomputed", "_reported"))
    for c in ("delta", "ci_lo", "ci_hi"):
        rows.append({"horizon": "all", "artifact": f"incremental_value.{c}", "n": len(merged),
                     "max_abs_diff": float((merged[f"{c}_recomputed"] - merged[f"{c}_reported"]).abs().max()),
                     "mismatches": 0})
    chosen = pd.read_csv(S["out"] / "D3_downstream_k2_samplers.csv")
    ref_s = pd.read_csv(config.RESULTS_DIR / "sampler_choice.csv")
    cmp = chosen.merge(ref_s, on=["horizon", "feature_set", "model"],
                       suffixes=("_recomputed", "_reported"))
    rows.append({"horizon": "all", "artifact": "sampler_choice", "n": len(cmp),
                 "max_abs_diff": 0.0,
                 "mismatches": int((cmp.sampler_recomputed != cmp.sampler_reported).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(S["out"] / "D3_k2_reproduction_check.csv", index=False)
    log("reproduction check:\n" + table.to_string())


# ---------------------------------------------------------------------------
# Derived outputs: summary tables, LaTeX bodies, figures
# ---------------------------------------------------------------------------

def _maybe(name):
    p = S["out"] / name
    return pd.read_csv(p) if p.exists() else None


def _reference():
    inc = pd.read_csv(config.RESULTS_DIR / "incremental_value.csv")
    inc = inc[inc.comparison_role == "primary"][
        ["horizon", "comparison", "metric", "delta", "ci_lo", "ci_hi", "p_bootstrap"]].copy()
    inc["se_implied"] = (inc.ci_hi - inc.ci_lo) / 3.92
    sv = pd.read_csv(config.RESULTS_DIR / "survival_cindex_delta.csv")
    sv = sv[sv.thyroid_set == THY][["model", "delta_c_index", "ci_lo", "ci_hi", "p_bootstrap"]]
    cif = pd.read_csv(config.RESULTS_DIR / "competing_cif_delta.csv")
    cif = cif[cif.thyroid_set == THY][["horizon", "delta_c_index", "ci_lo", "ci_hi", "p_bootstrap"]]
    return inc, sv, cif


def partition_summary():
    """Phase A: distribution of the paired delta across partitions."""
    d = _maybe("A_partitions_deltas.csv")
    if d is None:
        return None
    inc, _, _ = _reference()
    g = d.groupby(["horizon", "comparison", "metric"])
    s = g["delta"].agg(n="count", mean="mean", sd="std",
                       p2_5=lambda v: np.percentile(v, 2.5),
                       p97_5=lambda v: np.percentile(v, 97.5)).reset_index()
    s["share_pos"] = g["delta"].apply(lambda v: float((v > 0).mean())).to_numpy()
    s["share_ci_excl0"] = g["excludes_zero"].mean().to_numpy()
    s = s.merge(inc.rename(columns={"delta": "reported_delta", "ci_lo": "reported_lo",
                                    "ci_hi": "reported_hi", "se_implied": "reported_se"}),
                how="left")
    s["sd_over_reported_se"] = s["sd"] / s["reported_se"]
    s["reported_pct"] = [
        float((d[(d.horizon == r.horizon) & (d.comparison == r.comparison)
                 & (d.metric == r.metric)]["delta"] <= r.reported_delta).mean())
        for r in s.itertuples()]
    return s


def survival_partition_summary():
    """Phases B1, B2, B5: dependence of the delta C on the fold partition."""
    _, sv, cif = _reference()
    blocks = []
    for name, label, ref in [("B1_cox_partitions.csv", "Analysis B, Cox", sv[sv.model == "Cox"]),
                             ("B5_rsf_default_partitions.csv", "Analysis B, RSF (default configuration)",
                              sv[sv.model == "RSF"])]:
        t = _maybe(name)
        if t is not None:
            blocks.append(_seed_block(t, label, float(ref.delta_c_index.iloc[0]),
                                      float(ref.ci_lo.iloc[0]), float(ref.ci_hi.iloc[0])))
    t = _maybe("B2_cif_partitions.csv")
    if t is not None:
        for h, g in t.groupby("horizon"):
            ref = cif[cif.horizon == h]
            blocks.append(_seed_block(g, f"Analysis C, {h:g} years", float(ref.delta_c_index.iloc[0]),
                                      float(ref.ci_lo.iloc[0]), float(ref.ci_hi.iloc[0])))
    return pd.DataFrame(blocks) if blocks else None


def _seed_block(t, label, reported, lo, hi):
    v = t["delta_c_index"]
    within_se = float(((t["ci_hi"] - t["ci_lo"]) / 3.92).mean())
    return {"analysis": label, "n_partitions": int(len(t)), "mean": float(v.mean()),
            "sd_across": float(v.std()), "within_se": within_se,
            "sd_over_within_se": float(v.std() / within_se),
            "share_pos": float((v > 0).mean()), "share_ci_excl0": float(t["excludes_zero"].mean()),
            "reported": reported, "reported_lo": lo, "reported_hi": hi}


def refit_summary():
    """Phases B3, B4: out-of-bag refit bootstrap against the no-refit interval."""
    _, sv, cif = _reference()
    rows = []
    t = _maybe("B3_oob_refit_cox.csv")
    if t is not None:
        ref = sv[sv.model == "Cox"].iloc[0]
        rows.append(_refit_block(t.dropna(subset=["delta"]), "Analysis B, Cox", ref))
    t = _maybe("B4_oob_refit_cif.csv")
    if t is not None:
        for h, g in t.dropna(subset=["delta"]).groupby("horizon"):
            ref = cif[cif.horizon == h].iloc[0]
            rows.append(_refit_block(g, f"Analysis C, {h:g} years", ref))
    return pd.DataFrame(rows) if rows else None


def _refit_block(g, label, ref):
    v = g["delta"]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"analysis": label, "n_draws": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
            "p2_5": float(lo), "p97_5": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))),
            "mean_n_oob": float(g["n_oob"].mean()),
            "reported": float(ref.delta_c_index), "reported_lo": float(ref.ci_lo),
            "reported_hi": float(ref.ci_hi),
            # Widening is the ratio of the 2.5th-97.5th percentile width of the
            # refit distribution to the width of the reported interval; the SD
            # ratio compares the refit SD with the SE implied by that interval.
            "width_ratio": float((hi - lo) / (ref.ci_hi - ref.ci_lo)),
            "sd_ratio": float(v.std() / ((ref.ci_hi - ref.ci_lo) / 3.92))}


def yardstick_summary():
    """Phase D2: winners of each inner-fold count on the training-only yardstick."""
    t = _maybe("D2_training_yardstick.csv")
    if t is None:
        return None
    g = t.groupby(["horizon", "feature_set", "model", "k"])[["f1_macro", "roc_auc"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    piv = t.pivot_table(index=["horizon", "feature_set", "model", "rep_fold"], columns="k",
                        values="f1_macro")
    diffs = []
    for k in [c for c in piv.columns if c != 2]:
        dd = (piv[k] - piv[2]).groupby(level=[0, 1, 2]).agg(["mean", "std"]).reset_index()
        dd["k"] = k
        diffs.append(dd.rename(columns={"mean": "f1_diff_vs_k2", "std": "f1_diff_vs_k2_sd"}))
    if diffs:
        g = g.merge(pd.concat(diffs), how="left")
    return g


def tuning_summary():
    """Phase D1: winners per inner-fold count against the reported two-fold winners."""
    ref = pd.read_csv(config.RESULTS_DIR / "hyperparameter_search.csv")
    ref = ref[(ref.model.isin(MEMBERS)) & (ref.feature_set.isin([BASE, THY]))]
    ref = ref.assign(k=2)[["k", "horizon", "feature_set", "model", "best_cv_f1_macro",
                           "elapsed_seconds", "n_failed_candidates", "best_params"]].copy()
    ref["best_params"] = ref["best_params"].map(lambda s: json.dumps(json.loads(s), sort_keys=True))
    blocks = [ref]
    for k in S["d_ks"]:
        t, p = _maybe(f"D1_tuning_k{k}.csv"), S["out"] / f"D1_params_k{k}.json"
        if t is None or not p.exists():
            continue
        params = json.loads(p.read_text())
        t = t[["k", "horizon", "feature_set", "model", "best_cv_f1_macro", "elapsed_seconds",
               "n_failed_candidates"]].copy()
        t["best_params"] = [json.dumps(params[_tune_key(r.horizon, r.feature_set, r.model)], sort_keys=True)
                            for r in t.itertuples()]
        blocks.append(t)
    allt = pd.concat(blocks, ignore_index=True).sort_values(["horizon", "feature_set", "model", "k"])
    k2 = allt[allt.k == 2].set_index(["horizon", "feature_set", "model"])["best_params"]
    allt["same_as_k2"] = [k2.get((r.horizon, r.feature_set, r.model)) == r.best_params
                          for r in allt.itertuples()]
    return allt.reset_index(drop=True)


def test_sensitivity_summary():
    """Phase D3: the frozen-test result under each inner-fold count."""
    blocks = []
    for k in (2, *S["d_ks"]):
        t = _maybe(f"D3_downstream_k{k}_deltas.csv")
        if t is not None:
            blocks.append(t.assign(k=k))
    if not blocks:
        return None
    t = pd.concat(blocks, ignore_index=True)
    return t[["k", "horizon", "comparison", "metric", "delta", "ci_lo", "ci_hi", "p_bootstrap",
              "excludes_zero", "threshold_base", "threshold_other"]]


def _md(df, floatfmt=4):
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_summary():
    parts = ["# Robustness analyses\n",
             "Post hoc analyses of the frozen results; nothing here alters the main analysis. "
             "Produced by `7_robustness.py`; see the module docstring for the design of each phase.\n"]
    status = S["out"] / "status.json"
    if status.exists():
        parts.append("## Phase status\n")
        parts.append(_md(pd.DataFrame([{"phase": k, **v} for k, v in json.loads(status.read_text()).items()])))
    c = _maybe("D3_k2_reproduction_check.csv")
    if c is not None:
        parts.append("## Reproduction gate (k = 2 flow against the frozen artifacts)\n")
        parts.append(_md(c, 8))
    for title, table, note in [
        ("A. Repeated partitions of the fixed-horizon analysis", partition_summary(),
         "`reported_pct` is the share of partitions with a delta at or below the frozen-test value."),
        ("B1, B2, B5. Survival analyses: dependence on the fold partition", survival_partition_summary(), ""),
        ("B3, B4. Bootstrap with model refitting (out-of-bag evaluation)", refit_summary(), ""),
        ("D1. Search winners per inner-fold count", tuning_summary(), ""),
        ("D2. Training-only yardstick (5 x 3-fold repeated CV, macro-F1 at 0.5)", yardstick_summary(), ""),
        ("D3. Frozen-test result per inner-fold count (a second reading of the test partition)",
         test_sensitivity_summary(), ""),
    ]:
        parts.append(f"## {title}\n")
        if note:
            parts.append(note + "\n")
        parts.append(_md(table))
    path = S["out"] / "SUMMARY.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    log(f"summary written to {path}")


def _num(v, digits=4, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    text = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return f"${text}$"


def _ci(lo, hi, digits=4):
    return f"$({lo:.{digits}f}, {hi:.{digits}f})$"


def _pct(v):
    return f"{100 * v:.0f}"


# The precision-recall summary is average precision (scikit-learn's
# average_precision_score), hence the label AP rather than an area.
METRIC_LABEL = {"roc_auc": "AUROC", "auprc": "AP", "f1_macro": "F1-macro", "brier": "Brier"}
METRIC_ORDER = ["roc_auc", "auprc", "f1_macro", "brier"]
MODE_LABEL = {"locked pipeline": "Locked pipeline",
              "independently optimized": "Independently optimised"}


def write_tables():
    """LaTeX table bodies for the thesis appendix, one file with all of them."""
    out = []
    s = partition_summary()
    if s is not None:
        rows = []
        for mode in ("locked pipeline", "independently optimized"):
            rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{MODE_LABEL[mode]}}}}} \\\\")
            for h in S["horizons"]:
                for i, m in enumerate(METRIC_ORDER):
                    r = s[(s.horizon == h) & (s.comparison == mode) & (s.metric == m)].iloc[0]
                    first = f"{h} years" if i == 0 else ""
                    rows.append(f"{first} & {METRIC_LABEL[m]} & {_num(r['mean'])} ({r['sd']:.4f}) & "
                                f"{_ci(r.p2_5, r.p97_5)} & {_pct(r.share_pos)} & {_pct(r.share_ci_excl0)} & "
                                f"{_num(r.reported_delta)} [{_pct(r.reported_pct)}] \\\\")
            rows.append("\\midrule")
        rows.pop()
        out.append("% Table: repeated partitions (phase A)\n" + "\n".join(rows) + "\n")
    s = survival_partition_summary()
    if s is not None:
        rows = [f"{r.analysis.replace(' configuration', '')} & {_num(r['mean'])} & "
                f"{r.sd_across:.4f} & {r.within_se:.4f} & {r.sd_over_within_se:.2f} & "
                f"{_pct(r.share_pos)} & {_pct(r.share_ci_excl0)} & {_num(r.reported)} \\\\"
                for _, r in s.iterrows()]
        out.append("% Table: survival fold partitions (phases B1, B2, B5)\n" + "\n".join(rows) + "\n")
    s = refit_summary()
    if s is not None:
        rows = [f"{r.analysis} & {_num(r.reported)} {_ci(r.reported_lo, r.reported_hi)} & "
                f"{_num(r['mean'])} {_ci(r.p2_5, r.p97_5)} & {r.p_two_sided:.2f} & "
                f"{r.width_ratio:.2f} & {r.sd_ratio:.2f} \\\\"
                for _, r in s.iterrows()]
        out.append("% Table: refit bootstrap (phases B3, B4)\n" + "\n".join(rows) + "\n")
    s = yardstick_summary()
    if s is not None:
        rows = []
        e = s[s.model == "paper_ensemble"]
        for h in S["horizons"]:
            for fs in (BASE, THY):
                for i, k in enumerate((2, *S["d_ks"])):
                    r = e[(e.horizon == h) & (e.feature_set == fs) & (e.k == k)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    first = f"{h} years" if i == 0 else ""
                    fsname = "\\textsf{" + fs.replace("_", "\\_") + "}" if i == 0 else ""
                    diff = "--" if k == 2 else _num(r.f1_diff_vs_k2)
                    rows.append(f"{first} & {fsname} & {k} & {r.f1_macro_mean:.4f} ({r.f1_macro_std:.4f}) & "
                                f"{diff} & {r.roc_auc_mean:.4f} ({r.roc_auc_std:.4f}) \\\\")
                rows.append("\\midrule")
        rows.pop()
        out.append("% Table: training-only yardstick (phase D2)\n" + "\n".join(rows) + "\n")
    s = test_sensitivity_summary()
    if s is not None:
        rows = []
        for h in S["horizons"]:
            for m in METRIC_ORDER:
                for i, k in enumerate((2, *S["d_ks"])):
                    cells = []
                    for mode in ("locked pipeline", "independently optimized"):
                        r = s[(s.horizon == h) & (s.metric == m) & (s.k == k) & (s.comparison == mode)]
                        if r.empty:
                            cells.append("--")
                            continue
                        r = r.iloc[0]
                        star = "^{*}" if r.excludes_zero else ""
                        cells.append(f"{_num(r.delta)} $({r.ci_lo:.4f}, {r.ci_hi:.4f}){star}$")
                    first = f"{h} years" if (i == 0 and m == METRIC_ORDER[0]) else ""
                    label = METRIC_LABEL[m] if i == 0 else ""
                    rows.append(f"{first} & {label} & {k} & {cells[0]} & {cells[1]} \\\\")
            rows.append("\\midrule")
        rows.pop()
        out.append("% Table: frozen-test sensitivity (phase D3)\n" + "\n".join(rows) + "\n")
    path = S["out"] / "appendix_tables.tex"
    path.write_text("\n".join(out), encoding="utf-8")
    log(f"LaTeX table bodies written to {path}")


FIG_FONT = 10  # base font size; figures are sized so that this is about 9 pt in the thesis


def figure_style(plt):
    """Common style of the thesis figures: sans-serif, recessive axes and grid."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FIG_FONT,
        "axes.titlesize": FIG_FONT + 1, "axes.labelsize": FIG_FONT,
        "xtick.labelsize": FIG_FONT - 1, "ytick.labelsize": FIG_FONT - 1,
        "legend.fontsize": FIG_FONT - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "0.9",
        "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.edgecolor": "0.3", "xtick.color": "0.3", "ytick.color": "0.3",
        "savefig.dpi": 200, "figure.dpi": 100,
    })


def save_png(fig, name, dpi=200):
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


COLOURS = {"locked pipeline": "tab:blue", "independently optimized": "tab:orange"}


def _partition_figure(plt, d, inc, h, rng):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.2))
    for ax, m in zip(axes.ravel(), METRIC_ORDER):
        for x, mode in enumerate(COLOURS):
            v = d[(d.horizon == h) & (d.metric == m) & (d.comparison == mode)]["delta"].to_numpy()
            ax.scatter(x - 0.14 + rng.uniform(-0.07, 0.07, len(v)), v, s=16, alpha=0.5,
                       color=COLOURS[mode], linewidths=0, label=MODE_LABEL[mode])
            ax.hlines(v.mean(), x - 0.30, x + 0.02, color=COLOURS[mode], linewidth=2.0,
                      label="mean across partitions" if x == 0 else None)
            r = inc[(inc.horizon == h) & (inc.metric == m) & (inc.comparison == mode)].iloc[0]
            ax.errorbar(x + 0.22, r.delta, yerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]],
                        fmt="o", color="black", markersize=4.5, capsize=3, linewidth=1.2,
                        label="frozen test, 95% CI" if x == 0 else None)
        ax.axhline(0, color="0.45", linewidth=0.9, linestyle=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["locked", "independent"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(METRIC_LABEL[m])
        ax.set_ylabel("difference, thyroid minus baseline")
    handles, labels = axes[0][0].get_legend_handles_labels()
    keep = {}
    for hd, lb in zip(handles, labels):
        keep.setdefault(lb, hd)
    fig.legend(list(keep.values()), list(keep.keys()), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"{h}-year horizon: 50 alternative partitions", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return fig


def _seed_panel(ax, g, ref, title):
    x = np.arange(len(g))
    ax.errorbar(x, g["delta_c_index"], yerr=[g["delta_c_index"] - g["ci_lo"],
                                             g["ci_hi"] - g["delta_c_index"]],
                fmt="o", color="tab:blue", markersize=4.5, capsize=3, linewidth=1.2,
                label="fold assignment, 95% paired bootstrap interval")
    ax.axhline(ref.delta_c_index, color="black", linewidth=1.2, linestyle="--",
               label="reported estimate")
    ax.axhline(0, color="0.45", linewidth=0.9, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in g["seed"]])
    ax.set_xlabel("fold-assignment seed")
    ax.set_title(title)


def _survival_figure(plt, panels, name):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
    for ax, (title, g, ref) in zip(axes, panels):
        _seed_panel(ax, g, ref, title)
    axes[0].set_ylabel("$\Delta$C, thyroid minus baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_png(fig, name)
    plt.close(fig)


def _refit_figure(plt, s):
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ys = np.arange(len(s))[::-1]
    for i, (_, r) in enumerate(s.iterrows()):
        y = ys[i]
        ax.errorbar(r.reported, y + 0.16, xerr=[[r.reported - r.reported_lo], [r.reported_hi - r.reported]],
                    fmt="o", color="0.35", markersize=4.5, capsize=3, linewidth=1.2,
                    label="reported, no refit" if i == 0 else None)
        ax.errorbar(r["mean"], y - 0.16, xerr=[[r["mean"] - r.p2_5], [r.p97_5 - r["mean"]]],
                    fmt="s", color="tab:blue", markersize=4.5, capsize=3, linewidth=1.2,
                    label="refit, out-of-bag" if i == 0 else None)
    ax.axvline(0, color="0.45", linewidth=0.9, linestyle=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(s["analysis"])
    ax.set_xlabel("$\Delta$C, thyroid minus baseline, with 95% interval")
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def write_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_style(plt)

    inc, sv, cif = _reference()
    d = _maybe("A_partitions_deltas.csv")
    if d is not None:
        for h in S["horizons"]:
            fig = _partition_figure(plt, d, inc, h, np.random.default_rng(SEED))
            save_png(fig, f"robustness_partitions_h{h}")
            plt.close(fig)

    cox, rsf = _maybe("B1_cox_partitions.csv"), _maybe("B5_rsf_default_partitions.csv")
    if cox is not None and rsf is not None:
        _survival_figure(plt, [("Cox reference", cox, sv[sv.model == "Cox"].iloc[0]),
                               ("Random survival forest (default configuration)", rsf,
                                sv[sv.model == "RSF"].iloc[0])],
                         "robustness_survival_analysis_b")
    t = _maybe("B2_cif_partitions.csv")
    if t is not None:
        panels = [(f"{h:g}-year horizon", g.reset_index(drop=True), cif[cif.horizon == h].iloc[0])
                  for h, g in t.groupby("horizon")]
        _survival_figure(plt, panels, "robustness_survival_analysis_c")

    s = refit_summary()
    if s is not None:
        fig = _refit_figure(plt, s)
        save_png(fig, "robustness_refit_intervals")
        plt.close(fig)
    log("figures written")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def phases():
    return [
        ("D3_k2_check", lambda: phase_d_downstream(2)),
        ("A", phase_a),
        ("B1_cox", phase_b_cox),
        ("B2_cif", phase_b_cif),
        ("B3_oob_cox", lambda: phase_b_oob("coxB")),
        ("B4_oob_cif", lambda: phase_b_oob("cifC")),
        ("D1_k3", lambda: phase_d_tune(3)),
        ("D3_k3", lambda: phase_d_downstream(3)),
        ("B5_rsf", phase_b_rsf),
        ("D1_k5", lambda: phase_d_tune(5)),
        ("D2_yardstick", phase_d_yardstick),
        ("D3_k5", lambda: phase_d_downstream(5)),
    ]


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    only = [a for a in argv if not a.startswith("--")]
    smoke = "--smoke" in flags
    derived = {"--summary", "--tables", "--figures"} & flags
    if smoke:
        S.update(out=config.RESULTS_DIR / "robustness_smoke", horizons=(7,),
                 a_reps=1, a_boot=20, a_workers=5, b_seeds=[SEED], b_boot=20, b_workers=1,
                 oob_boot=8, oob_workers=4, d_iter=4, d_ks=(3,), yard_reps=1, yard_k=3,
                 down_boot=20, rsf_trees=20)
    S["out"].mkdir(parents=True, exist_ok=True)
    run_phases = bool(only) or not derived
    if run_phases:
        status_path = S["out"] / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        sizes = {k: (str(v) if isinstance(v, Path) else v) for k, v in S.items()}
        log(f"run start smoke={smoke} sizes={sizes}")
        for name, fn in phases():
            if only and name not in only:
                continue
            started = time.time()
            log(f"=== {name} start ===")
            try:
                fn()
                status[name] = {"ok": True, "elapsed_s": round(time.time() - started)}
            except Exception:
                log(f"!!! {name} failed:\n{traceback.format_exc()}")
                status[name] = {"ok": False, "elapsed_s": round(time.time() - started)}
            status_path.write_text(json.dumps(status, indent=1))
            log(f"=== {name} end ({status[name]['elapsed_s']} s) ===")
        log("run complete")
    if "--summary" in flags or not derived:
        write_summary()
    if smoke:
        # The smoke profile validates the code paths only; its tables and
        # figures must never replace the reported ones.
        return
    if "--tables" in flags or not derived:
        write_tables()
    if "--figures" in flags or not derived:
        write_figures()


if __name__ == "__main__":
    main(sys.argv[1:])
