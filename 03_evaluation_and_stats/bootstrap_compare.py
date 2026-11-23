"""Fast exact vectorized paired account-cluster bootstrap matching scripts/statistical_analysis.compare."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "scripts"), str(ROOT / "work" / "antigravity_three_experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common_eval as ce
from decompose_variance import components
from evaluate_stability import account_metrics
from experiment_common import prediction_data, probability, write_json

OUT_ROOT = ROOT / "work" / "antigravity_robust_mh5"


def precompute_model_arrays(pred_dict, thresholds):
    logits = pred_dict["logits"]
    y = pred_dict["labels"]
    thr = np.asarray(thresholds, dtype=np.float64)
    metrics = account_metrics(logits, y, thr)
    comp = components(logits)[:, :2]  # [N, 2]
    p = probability(logits).reshape(len(y), -1)  # [N, R*M]
    order = np.argsort(-p, axis=0, kind="mergesort")
    y_sorted = y[order].astype(np.float64)  # [N, K]
    return {
        "y": y,
        "order": order,
        "y_sorted": y_sorted,
        "churn": metrics["churn"].mean(axis=1),  # [N]
        "stable_error": metrics["stable_error"].mean(axis=1),  # [N]
        "logit_var": metrics["logit_variance"].mean(axis=1),  # [N]
        "tr_var": comp[:, 0],  # [N]
        "sp_var": comp[:, 1],  # [N]
    }


def fast_ap_mean(order, y_sorted, weights):
    # Exact tie-grouped sklearn average_precision_score across all R*M columns simultaneously
    w_sorted = weights[order]  # [N, K]
    pos_w_sorted = y_sorted * w_sorted
    tp = np.cumsum(pos_w_sorted, axis=0)
    tot = np.cumsum(w_sorted, axis=0)
    pos_total = tp[-1]  # [K]
    # Exact step integral: sum over items where y_i==1 and w_i>0 of (w_i * tp / tot) / pos_total
    prec = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
    aps = (prec * pos_w_sorted).sum(axis=0) / pos_total
    return float(aps.mean())


def fast_estimates(pre, weights):
    w = weights
    y = pre["y"]
    w_sum = w.sum()
    w1 = w * (y == 1)
    w0 = w * (y == 0)
    ap = fast_ap_mean(pre["order"], pre["y_sorted"], w)
    churn = float((pre["churn"] * w).sum() / w_sum)
    bm = float((pre["stable_error"] * w1).sum() / w1.sum())
    hfp = float((pre["stable_error"] * w0).sum() / w0.sum())
    lvar = float((pre["logit_var"] * w).sum() / w_sum)
    trv = float((pre["tr_var"] * w).sum() / w_sum)
    spv = float((pre["sp_var"] * w).sum() / w_sum)
    return np.array([ap, churn, bm, hfp, lvar, trv, spv], dtype=np.float64)


def compare_fast(ref_pred, ref_thr, cand_pred, cand_thr, iterations=1000, seed=20261118):
    for key in ("ids", "labels", "seeds", "library_sha256", "view_slots"):
        np.testing.assert_array_equal(ref_pred[key], cand_pred[key])
    labels = ref_pred["labels"]
    pre_ref = precompute_model_arrays(ref_pred, ref_thr)
    pre_cand = precompute_model_arrays(cand_pred, cand_thr)

    w_ones = np.ones(len(labels), dtype=np.float64)
    measured = fast_estimates(pre_cand, w_ones) - fast_estimates(pre_ref, w_ones)

    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == c) for c in (0, 1)]
    samples = np.empty((iterations, len(measured)), dtype=np.float64)
    for b in range(iterations):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        w = np.bincount(ids, minlength=len(labels)).astype(np.float64)
        samples[b] = fast_estimates(pre_cand, w) - fast_estimates(pre_ref, w)

    names = [
        "ap", "churn", "bot_stable_miss", "human_stable_false_positive",
        "within_account_logit_variance", "training_variance", "sampling_variance",
    ]
    rows = {}
    for j, name in enumerate(names):
        centered = samples[:, j] - measured[j]
        p = (1 + np.count_nonzero(np.abs(centered) >= abs(measured[j]))) / (iterations + 1)
        rows[name] = {
            "effect": float(measured[j]),
            "ci95": np.quantile(samples[:, j], [0.025, 0.975]).tolist(),
            "p_two_sided": float(p),
        }
    valid = sorted(names, key=lambda n: rows[n]["p_two_sided"])
    adjusted = 0.0
    for rank, name in enumerate(valid):
        adjusted = max(adjusted, min(1.0, (len(valid) - rank) * rows[name]["p_two_sided"]))
        rows[name]["p_holm"] = adjusted
    rows["ap"]["noninferiority_margin"] = 0.001
    rows["ap"]["one_sided_lower95"] = float(np.quantile(samples[:, 0], 0.05))
    rows["ap"]["noninferior"] = rows["ap"]["one_sided_lower95"] > -0.001
    rows["ap"]["p_noninferiority_unadjusted"] = float(
        (1 + np.count_nonzero(samples[:, 0] - measured[0] >= measured[0] + 0.001)) / (iterations + 1)
    )
    return rows


def main():
    t0 = time.perf_counter()
    analysis = json.loads((OUT_ROOT / "robust_analysis.json").read_text(encoding="utf8"))
    ref_paths = {
        "EXP-0": ce.FORMAL_FULL_ROOT / "EXP-0" / "predictions.npz",
        "EXP-8-MH5-KL": ce.EXP_ROOT / "exp2_ensemble" / "full" / "EXP-8-MH5-KL" / "predictions.npz",
    }
    cand_names = ["K4-MH5-KL01", "K4-MH5-SetAttn", "K4-MH5-SubBag"]
    refs = {k: prediction_data(v) for k, v in ref_paths.items()}
    cands = {n: prediction_data(OUT_ROOT / "runs" / "full" / n / "predictions.npz") for n in cand_names}

    results = []
    for cname, cdata in cands.items():
        for rname, rdata in refs.items():
            for mode in ("cal_matched_human_fp", "dev_iso_human_stable_fp"):
                r_cuts = analysis["methods"][rname]["modes"][mode]["thresholds"]
                c_cuts = analysis["methods"][cname]["modes"][mode]["thresholds"]
                rows = compare_fast(rdata, r_cuts, cdata, c_cuts, 1000, 20261118)
                results.append({
                    "candidate": cname,
                    "reference": rname,
                    "calibration_mode": mode,
                    "metrics": rows,
                })
                ap, ch = rows["ap"], rows["churn"]
                bm, hfp = rows["bot_stable_miss"], rows["human_stable_false_positive"]
                spv = rows["sampling_variance"]
                print(
                    f"{cname:15s} vs {rname:12s} ({mode:23s}): "
                    f"dAP={ap['effect']:+.5f} [{ap['ci95'][0]:+.5f},{ap['ci95'][1]:+.5f}] p={ap['p_holm']:.4f} | "
                    f"dCh={ch['effect']*100:+.3f}pp p={ch['p_holm']:.4f} | "
                    f"dBM={bm['effect']*100:+.3f}pp p={bm['p_holm']:.4f} | "
                    f"dHFP={hfp['effect']*100:+.3f}pp p={hfp['p_holm']:.4f} | "
                    f"dSpV={spv['effect']:+.4f} p={spv['p_holm']:.4f}",
                    flush=True,
                )

    write_json(OUT_ROOT / "bootstrap_summary.json", {
        "iterations": 1000,
        "seed": 20261118,
        "elapsed_seconds": time.perf_counter() - t0,
        "comparisons": results,
        "test_scored": False,
    })
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
