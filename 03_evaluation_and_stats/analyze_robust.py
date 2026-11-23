"""Evaluate all antigravity_robust_mh5 full runs under exact unified protocol."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "work"))
sys.path.insert(0, str(ROOT / "work" / "antigravity_three_experiments"))

import common_eval as ce
from experiment_common import probability, write_json

OUT_ROOT = ROOT / "work" / "antigravity_robust_mh5"
RUNS_FULL = OUT_ROOT / "runs" / "full"


def main():
    proto = ce.load_protocol_indices()
    counts_dev = proto["counts_dev_primary"]

    exp0_dir = ce.FORMAL_FULL_ROOT / "EXP-0"
    exp0_pred = ce.prediction_data(exp0_dir / "predictions.npz")
    exp0_cuts = exp0_pred["thresholds"]
    exp0_cal_fps = []
    for r, s in enumerate(exp0_pred["seeds"].tolist()):
        with np.load(exp0_dir / f"seed_{int(s)}.npz") as sz:
            exp0_cal_fps.append(ce.compute_cal_human_fp_rate(sz["cal_y"], sz["cal_logits"][:, 0], float(exp0_cuts[r])))
    exp0_main, _ = ce.evaluate(exp0_pred["logits"], exp0_pred["labels"], exp0_cuts)
    ref_dev_human_stable_fp = exp0_main["mean"]["human_stable_error"]
    p0 = probability(exp0_pred["logits"])
    ref_dev_human_mean_fp = float((p0[exp0_pred["labels"] == 0] >= exp0_cuts[None, :, None]).mean())

    candidates = [
        ("EXP-0", ce.FORMAL_FULL_ROOT / "EXP-0"),
        ("EXP-4-CE", ce.FORMAL_FULL_ROOT / "EXP-4-CE"),
        ("EXP-4", ce.FORMAL_FULL_ROOT / "EXP-4"),
        ("EXP-8-MH5-KL", ce.EXP_ROOT / "exp2_ensemble" / "full" / "EXP-8-MH5-KL"),
        ("EXP-8-MH5-KL05", ce.EXP_ROOT / "exp2_ensemble" / "full" / "EXP-8-MH5-KL05"),
    ]
    if RUNS_FULL.exists():
        for d in sorted(RUNS_FULL.iterdir()):
            if d.is_dir() and (d / "predictions.npz").exists():
                candidates.append((d.name, d))

    full_json = {}
    rows = []
    for name, mdir in candidates:
        res = ce.evaluate_all_calibrations(mdir, exp0_cal_fps, ref_dev_human_stable_fp, ref_dev_human_mean_fp, counts_dev)
        full_json[name] = res
        var_ests = res["variance"]["mean_signed_estimates"]
        for mode_name, mdata in res["modes"].items():
            m = mdata["mean"]
            rows.append({
                "method": name,
                "calibration_mode": mode_name,
                "ap": m["ap"],
                "bot_f1": m["bot_f1"],
                "churn": m["churn"],
                "bot_stable_miss": m["bot_stable_error"],
                "human_stable_fp": m["human_stable_error"],
                "mean_threshold": float(np.mean(mdata["thresholds"])),
                "training_var": var_ests[0],
                "sampling_var": var_ests[1],
                "interaction_var": var_ests[2],
            })

    # Joint screening against EXP-0, EXP-4, EXP-8-MH5-KL
    screens = []
    for name in full_json:
        if name in ("EXP-0", "EXP-4-CE", "EXP-4", "EXP-8-MH5-KL", "EXP-8-MH5-KL05"):
            continue
        for ref_name in ("EXP-0", "EXP-4", "EXP-8-MH5-KL"):
            for mode in ("cal_matched_human_fp", "cal_v0_botf1"):
                c = full_json[name]["modes"][mode]
                r = full_json[ref_name]["modes"][mode]
                scr = ce.joint_screen(c["per_seed"], r["per_seed"], c["mean"], r["mean"])
                screens.append({"candidate": name, "reference": ref_name, "calibration_mode": mode, **scr})

    ce.write_csv(OUT_ROOT / "robust_summary.csv", rows)
    write_json(OUT_ROOT / "robust_analysis.json", {"methods": full_json, "joint_screening": screens, "test_scored": False})

    for mode in ("cal_matched_human_fp", "cal_v0_botf1"):
        print(f"\n=== {mode} ===")
        for r in rows:
            if r["calibration_mode"] == mode:
                print(
                    f"{r['method']:22s} | AP={r['ap']:.5f} | F1={r['bot_f1']:.5f} | "
                    f"Churn={r['churn']*100:.3f}% | BotMiss={r['bot_stable_miss']*100:.3f}% | "
                    f"HumFP={r['human_stable_fp']*100:.3f}% | Thr={r['mean_threshold']:.3f} | "
                    f"TrVar={r['training_var']:.4f} | SpVar={r['sampling_var']:.4f}"
                )
    print("\n=== Joint Screening ===")
    for s in screens:
        md = s["mean_details"]
        print(f"{s['candidate']:22s} vs {s['reference']:12s} ({s['calibration_mode']:20s}): {s['passed_seeds']}/5, mean_pass={s['mean_passed']}, red={md['relative_churn_reduction']*100:+.2f}%, safe={md['two_stable_errors_not_increased']}")


if __name__ == "__main__":
    main()
