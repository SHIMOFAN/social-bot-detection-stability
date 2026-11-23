"""Shared evaluation, variance decomposition, calibration alignment, and bootstrap scaffold
for the three TwiBot-20 8-tweet experiments."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
WORK_DIR = ROOT / "work"
for p in (str(SCRIPTS_DIR), str(WORK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiment_common import probability, prediction_data, save_npz, sha, write_json
from evaluate_stability import account_metrics, evaluate
from decompose_variance import components, summarize as summarize_variance
from statistical_analysis import compare as bootstrap_compare, finalize_family
import train_twibot20_local as base

SEEDS = [42, 43, 44, 45, 46]
SAMPLING_SEED = 20261118
BUDGET = 8
VIEWS = 30
SAMPLING_LIBRARY = ROOT / "results" / "sampling_stability" / "dev_30_views.npz"
FORMAL_FULL_ROOT = ROOT / "results" / "formal_experiments" / "full"
EXP_ROOT = ROOT / "work" / "antigravity_three_experiments"


def load_protocol_indices():
    """Return exact fit (6651), cal (1176), dev primary (2240), dev bank (2240,30,8), cal_slots (1176,1,8), mu, sd."""
    raw = base.load_data("train")
    dev = base.load_data("dev")
    original = json.loads((base.OUTPUT / "training_protocol.json").read_text(encoding="utf8"))
    fit = np.asarray(original["fit_indices"])
    cal = np.asarray(original["cal_indices"])
    fit = fit[raw["count"][fit] >= BUDGET]
    cal = cal[raw["count"][cal] >= BUDGET]
    with np.load(SAMPLING_LIBRARY) as saved:
        library = dict(saved)
    np.testing.assert_array_equal(library["ids"], dev["ids"])
    lib_sha = sha(SAMPLING_LIBRARY)
    di = np.flatnonzero(library["primary"])
    bank = library["slots"][di, :VIEWS]
    assert (bank >= 0).all() and bank.shape == (len(di), VIEWS, BUDGET)
    mu = np.asarray(original["meta_mean"], dtype=np.float32)
    sd = np.asarray(original["meta_std"], dtype=np.float32)
    rng_cal = np.random.default_rng(SAMPLING_SEED + 3)
    cal_slots = np.full((len(cal), 1, BUDGET), -1, dtype=np.int16)
    for i, c in enumerate(raw["count"][cal]):
        cal_slots[i, 0, :min(int(c), BUDGET)] = rng_cal.choice(int(c), min(int(c), BUDGET), replace=False)
    return {
        "raw": raw,
        "dev": dev,
        "fit": fit,
        "cal": cal,
        "di": di,
        "bank": bank,
        "counts_dev_primary": dev["count"][di],
        "cal_slots": cal_slots,
        "mu": mu,
        "sd": sd,
        "library_sha256": lib_sha,
    }


def matched_cutoff_for_rate(scores, is_target_group, target_rate, prior_cut=0.5):
    """Find threshold on `scores[is_target_group]` whose >= rate matches `target_rate`."""
    grp = np.asarray(scores, dtype=np.float64)[is_target_group]
    target_count = int(round(target_rate * len(grp)))
    values, counts = np.unique(grp, return_counts=True)
    if len(values) == 1:
        return float(values[0]), int((grp >= values[0]).sum())
    midpoints = 0.5 * (values[:-1] + values[1:])
    cuts = np.r_[np.nextafter(values[-1], np.inf), midpoints, values[0]]
    positives = np.r_[0, np.cumsum(counts[::-1])[:-1][::-1], counts.sum()]
    best_cut, best_pos = min(
        zip(cuts, positives, strict=True),
        key=lambda pair: (abs(int(pair[1]) - target_count), abs(float(pair[0]) - prior_cut)),
    )
    return float(best_cut), int(best_pos)


def matched_stable_human_fp_cutoff_dev(logits_rm, labels, target_human_stable_fp, prior_cut=0.5):
    """Post-hoc diagnostic on dev: threshold aligning M-view Human stable FP rate to target."""
    p = probability(logits_rm)  # [N, M]
    human_min_p = p[labels == 0].min(axis=1)
    return matched_cutoff_for_rate(human_min_p, np.ones(len(human_min_p), dtype=bool), target_human_stable_fp, prior_cut)


def matched_mean_human_fp_cutoff_dev(logits_rm, labels, target_human_fp, prior_cut=0.5):
    """Post-hoc diagnostic on dev: threshold aligning M-view mean Human FP rate to target."""
    p = probability(logits_rm)[labels == 0].ravel()
    return matched_cutoff_for_rate(p, np.ones(len(p), dtype=bool), target_human_fp, prior_cut)


def evaluate_predictions_with_subgroups(logits, labels, thresholds, counts):
    """Evaluate primary (2240) + >8 (2233) + >=16 (2183) + ==8 (7) cohorts."""
    main_res, acct = evaluate(logits, labels, thresholds)
    subgroups = {}
    for name, mask in (
        ("all_primary", np.ones(len(counts), dtype=bool)),
        ("count_gt_8", counts > 8),
        ("count_ge_16", counts >= 16),
        ("count_eq_8", counts == 8),
    ):
        if mask.any():
            sub_res, _ = evaluate(logits[mask], labels[mask], thresholds)
            subgroups[name] = {"accounts": int(mask.sum()), "mean": sub_res["mean"], "per_seed": sub_res["per_seed"]}
    return main_res, acct, subgroups


def joint_screen(cand_per_seed, ref_per_seed, cand_mean, ref_mean):
    """Exact joint screening criteria from docs/experiment_protocol.md & scripts/summarize_experiments.py."""
    details = []
    for i, (c, r) in enumerate(zip(cand_per_seed, ref_per_seed, strict=True)):
        d_ap = c["ap"] - r["ap"]
        d_f1 = c["bot_f1"] - r["bot_f1"]
        red = 1.0 - c["churn"] / r["churn"] if r["churn"] > 0 else None
        safe = (c["bot_stable_error"] <= r["bot_stable_error"]) and (c["human_stable_error"] <= r["human_stable_error"])
        passed = (d_ap >= -0.001) and (d_f1 >= -0.002) and (red is not None) and (red >= 0.20) and safe
        details.append({
            "seed_index": i,
            "ap_difference": float(d_ap),
            "f1_difference": float(d_f1),
            "relative_churn_reduction": float(red) if red is not None else None,
            "two_stable_errors_not_increased": bool(safe),
            "passed": bool(passed),
        })
    mean_d_ap = cand_mean["ap"] - ref_mean["ap"]
    mean_d_f1 = cand_mean["bot_f1"] - ref_mean["bot_f1"]
    mean_red = 1.0 - cand_mean["churn"] / ref_mean["churn"] if ref_mean["churn"] > 0 else None
    mean_safe = (cand_mean["bot_stable_error"] <= ref_mean["bot_stable_error"]) and (
        cand_mean["human_stable_error"] <= ref_mean["human_stable_error"]
    )
    mean_passed = (mean_d_ap >= -0.001) and (mean_d_f1 >= -0.002) and (mean_red is not None) and (mean_red >= 0.20) and mean_safe
    passed_seeds = sum(int(d["passed"]) for d in details)
    return {
        "passed_seeds": passed_seeds,
        "seed_count": len(details),
        "mean_passed": bool(mean_passed),
        "joint_passed": bool(passed_seeds >= 4 and mean_passed) if len(details) >= 5 else False,
        "mean_details": {
            "ap_difference": float(mean_d_ap),
            "f1_difference": float(mean_d_f1),
            "relative_churn_reduction": float(mean_red) if mean_red is not None else None,
            "two_stable_errors_not_increased": bool(mean_safe),
        },
        "details": details,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_cal_human_fp_rate(cal_y, cal_logits_1d, cut):
    p = probability(cal_logits_1d)
    human = cal_y == 0
    return float((p[human] >= cut).mean())


def evaluate_all_calibrations(method_dir, ref_cal_human_fp_rates, ref_dev_human_stable_fp, ref_dev_human_fp, counts_dev):
    """Evaluate a method directory containing predictions.npz and seed_{s}.npz under 4 calibration modes."""
    pred = prediction_data(method_dir / "predictions.npz")
    logits = pred["logits"]  # [N, R, M]
    labels = pred["labels"]
    seeds = pred["seeds"].tolist()
    orig_cuts = pred["thresholds"]

    cal_matched_cuts = []
    dev_iso_stable_cuts = []
    dev_iso_mean_cuts = []
    cal_human_fps_orig = []
    cal_human_fps_matched = []

    for r, s in enumerate(seeds):
        seed_npz = method_dir / f"seed_{int(s)}.npz"
        if seed_npz.exists():
            with np.load(seed_npz) as sz:
                cal_y = sz["cal_y"]
                cal_logits = sz["cal_logits"][:, 0]
            cal_fp = compute_cal_human_fp_rate(cal_y, cal_logits, float(orig_cuts[r]))
            cal_human_fps_orig.append(cal_fp)
            target_fp = ref_cal_human_fp_rates[r] if r < len(ref_cal_human_fp_rates) else float(np.mean(ref_cal_human_fp_rates))
            mcut, mpos = matched_cutoff_for_rate(probability(cal_logits), cal_y == 0, target_fp, float(orig_cuts[r]))
            cal_matched_cuts.append(mcut)
            cal_human_fps_matched.append(mpos / int((cal_y == 0).sum()))
        else:
            cal_matched_cuts.append(float(orig_cuts[r]))

        dcut_s, _ = matched_stable_human_fp_cutoff_dev(logits[:, r, :], labels, ref_dev_human_stable_fp, float(orig_cuts[r]))
        dev_iso_stable_cuts.append(dcut_s)
        dcut_m, _ = matched_mean_human_fp_cutoff_dev(logits[:, r, :], labels, ref_dev_human_fp, float(orig_cuts[r]))
        dev_iso_mean_cuts.append(dcut_m)

    modes = {
        "cal_v0_botf1": np.asarray(orig_cuts, dtype=np.float64),
        "cal_matched_human_fp": np.asarray(cal_matched_cuts, dtype=np.float64),
        "dev_iso_human_stable_fp": np.asarray(dev_iso_stable_cuts, dtype=np.float64),
        "dev_iso_human_mean_fp": np.asarray(dev_iso_mean_cuts, dtype=np.float64),
    }
    results = {}
    for mode_name, cuts in modes.items():
        main_res, _, subgroups = evaluate_predictions_with_subgroups(logits, labels, cuts, counts_dev)
        results[mode_name] = {
            "thresholds": cuts.tolist(),
            "mean": main_res["mean"],
            "per_seed": main_res["per_seed"],
            "count_gt_8_mean": subgroups["count_gt_8"]["mean"],
            "count_ge_16_mean": subgroups["count_ge_16"]["mean"],
        }
    var_res, _ = summarize_variance(logits)
    return {
        "method": str(pred["method"]),
        "seeds": seeds,
        "cal_human_fp_orig": cal_human_fps_orig,
        "cal_human_fp_matched": cal_human_fps_matched,
        "variance": var_res,
        "modes": results,
    }
