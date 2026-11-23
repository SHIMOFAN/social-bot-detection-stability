"""K-layer (K=4) partial-unfreeze runner reusing sub-bag + clean MH5-KL objectives.

Supports both:
- objective_type: "subbag_probvar" (S=8 sub-bags from 16 tweets + prob_var + anti-collapse)
- objective_type: "mh5_kl" (clean 2-view dynamic MH5 + logit_kl control isolating K=4)
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import site
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(ROOT / "scripts"), str(ROOT / "work")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiment_common import read_config, save_npz, sha, write_json
from evaluate_stability import evaluate
import train_twibot20_local as base
from encode_own_tweets import Encoder
from prepare_twibot20_local import AUDIT
from train_twibot20_resampling_factorial import IndexedCache, gather_views
from run_experiment import objective, predict_bank, slots_for, stratified_limit
from run_robust_exp import RobustSharedHeads, sample_subbags_from_pool, subbag_objective


class LastKBlock(nn.Module):
    """Wraps the final K MiniLM BertLayer blocks behind BertLayer's (hidden, attention_mask) interface."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask=None):
        x = hidden_states
        for blk in self.layers:
            x = blk(x, attention_mask=attention_mask)[0]
        return (x,)


def run_single_config(cfg_path, stage):
    cfg = read_config(cfg_path)
    cfg.update(cfg["stages"][stage])
    k_layers = int(cfg["unfreeze_layers"])
    if not (1 <= k_layers <= 12):
        raise ValueError(f"Invalid unfreeze_layers={k_layers}")
    if cfg["budget"] != 8:
        raise ValueError("Budget must be 8")

    # Verify cache metadata matches k_layers (and test_opened is False)
    cache_dir = ROOT / cfg["prefix_cache"]
    proto = json.loads((cache_dir / "cache_protocol.json").read_text(encoding="utf8"))
    assert proto["test_opened"] is False
    if k_layers > 1:
        assert int(proto["k_layers"]) == k_layers, (proto, k_layers)

    out = ROOT / cfg["output_root"] / stage / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    resolved = dict(cfg)
    resolved["runner_sha256"] = sha(Path(__file__))
    config_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    existing = out / "resolved_config.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf8")) != resolved:
        raise ValueError(f"Existing config differs at {out}; use new name/output_root")
    write_json(existing, resolved)

    assert torch.cuda.is_available() and not site.ENABLE_USER_SITE
    torch.set_num_threads(cfg["cpu_threads"])
    sys.path.append(str(AUDIT / "encoder_deps"))
    encoder = Encoder(AUDIT / "encoder", cfg["sentence_microbatch"], 128, "cuda")
    template_layers = [deepcopy(b).float().eval() for b in encoder.model.encoder.layer[-k_layers:]]
    del encoder
    sys.path.append(site.getusersitepackages())
    torch.set_float32_matmul_precision("highest")

    import train_twibot20_resampling_factorial as cache_code
    cache_code.SENTENCES = cfg["sentence_microbatch"]

    raw, dev = base.load_data("train"), base.load_data("dev")
    original = json.loads((base.OUTPUT / "training_protocol.json").read_text(encoding="utf8"))
    fit, cal = (np.asarray(original[k]) for k in ("fit_indices", "cal_indices"))
    fit = fit[raw["count"][fit] >= 8]
    fit = stratified_limit(fit, raw["labels"], cfg["fit_limit"], cfg["sampling_seed"])
    cal = cal[raw["count"][cal] >= 8]
    cal = stratified_limit(cal, raw["labels"], cfg["cal_limit"], cfg["sampling_seed"] + 1)

    libpath = ROOT / cfg["sampling_library"]
    with np.load(libpath) as saved:
        library = dict(saved)
    np.testing.assert_array_equal(library["ids"], dev["ids"])
    library_hash = sha(libpath)
    di = np.flatnonzero(library["primary"])
    di = stratified_limit(di, dev["labels"], cfg["eval_limit"], cfg["sampling_seed"] + 2)
    bank = library["slots"][di, :cfg["views"]]
    assert (bank >= 0).all() and bank.shape[-1] == 8

    mu, sd = (np.asarray(original[k], dtype=np.float32) for k in ("meta_mean", "meta_std"))
    profiles = [base.make_views(d, ids, mu, sd) for d, ids in ((raw, fit), (raw, cal), (dev, di))]
    caches = [
        IndexedCache(cache_dir, split, d["ids"], d["count"], width)
        for split, d, width in (("train", raw, 16), ("dev", dev, 32))
    ]
    cal_slots = slots_for(raw["count"][cal], 1, np.random.default_rng(cfg["sampling_seed"] + 3), 8)

    fit_counts = raw["count"][fit]
    pool_slots_fit = np.full((len(fit), 16), -1, dtype=np.int16)
    for i, c in enumerate(fit_counts):
        k = min(int(c), 16)
        pool_slots_fit[i, :k] = np.arange(k, dtype=np.int16)

    scores, thresholds, seed_values, head_values = [], [], [], []
    arch = cfg.get("arch", "standard")
    obj_type = cfg.get("objective_type", "subbag_probvar")
    S_bags = int(cfg.get("subbags", 8))

    for seed in cfg["seeds"]:
        pending = out / f"seed_{seed}.npz"
        if all(p.exists() for p in (pending, out / f"seed_{seed}.json", out / f"seed_{seed}.pt")):
            with np.load(pending) as cached:
                scores.append(cached["logits"])
                thresholds.append(float(cached["threshold"]))
                head_values.append(cached["per_head"])
            seed_values.append(json.loads((out / f"seed_{seed}.json").read_text(encoding="utf8")))
            continue

        torch.manual_seed(seed)
        weights = torch.load(
            ROOT / cfg["initialization"] / f"seed_{seed}_tuned_natural.pt",
            map_location="cuda",
            weights_only=True,
        )
        layer = LastKBlock([deepcopy(b) for b in template_layers]).requires_grad_(True).eval().cuda()
        if cfg.get("init_last_layer", True):
            layer.layers[-1].load_state_dict(weights["last_layer"])
        anchor = base.AccountHead().cuda()
        anchor.load_state_dict(weights["head"])
        head = RobustSharedHeads(
            anchor,
            cfg["heads"],
            arch=arch,
            attn_temp=float(cfg.get("attn_temp", 1.0)),
            gate_init_tau=float(cfg.get("gate_init_tau", 0.55)),
        ).cuda()

        nparams = sum(p.numel() for p in layer.parameters() if p.requires_grad) + sum(
            p.numel() for p in head.parameters()
        )
        parameters = [p for p in list(layer.parameters()) + list(head.parameters()) if p.requires_grad]
        groups = [
            {"params": head.parameters(), "lr": cfg["head_lr"]},
            {"params": layer.parameters(), "lr": cfg["encoder_lr"]},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])

        rng = np.random.default_rng(cfg["sampling_seed"] + seed)
        fixed = slots_for(raw["count"][fit], cfg.get("train_views", 2), rng, 8)
        pos = torch.tensor(
            (raw["labels"][fit] == 0).sum() / (raw["labels"][fit] == 1).sum(),
            device="cuda",
            dtype=torch.float32,
        )
        cfg["_pos_weight"] = pos

        traces = []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        for epoch in range(cfg["epochs"]):
            head.train()
            plan = (
                slots_for(raw["count"][fit], cfg.get("train_views", 2), rng, 8)
                if cfg.get("dynamic", True) and epoch > 0
                else fixed
            )
            order = np.random.default_rng(seed + epoch * 1000).permutation(len(fit))
            losses = []
            for start in range(0, len(order), cfg["batch"]):
                idx = order[start:start + cfg["batch"]]
                if obj_type == "mh5_kl":
                    table = caches[0].selected(layer, fit[idx], plan[idx])
                    view = gather_views(table, plan[idx], profiles[0].at(idx))
                    z, pooled, _, _ = head.pieces(view)
                    cls_val, cons_val = objective(z, pooled, view.labels, cfg)
                    loss = cls_val + cfg["lambda_cons"] * cons_val
                elif obj_type == "subbag_probvar":
                    table = caches[0].selected(layer, fit[idx], pool_slots_fit[idx, None, :])
                    sub_slots = sample_subbags_from_pool(fit_counts[idx], S_bags, rng, budget=8, max_pool=16)
                    loss, cls_val, cons_val = subbag_objective(
                        head, table, fit_counts[idx], profiles[0].at(idx), sub_slots, cfg
                    )
                else:
                    raise ValueError(f"Unknown objective_type={obj_type}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append((float(loss.detach()), float(cls_val.detach()), float(cons_val.detach())))
            traces.append({"epoch": epoch + 1, "loss": np.mean(losses, axis=0).tolist(), "steps": len(losses)})

        head.eval()
        train_seconds = time.perf_counter() - started
        eval_started = time.perf_counter()

        calibration, _, _ = predict_bank(layer, head, caches[0], cal, cal_slots, profiles[1], cfg["batch"])
        cutoff = float(base.cutoff(raw["labels"][cal], calibration[:, 0]))
        pred, individual, ablated = predict_bank(layer, head, caches[1], di, bank, profiles[2], cfg["batch"], True)
        result, _ = evaluate(pred[:, None], dev["labels"][di], np.array([cutoff]))
        branch = {k: evaluate(v[:, None], dev["labels"][di], np.array([cutoff]))[0]["mean"] for k, v in ablated.items()}

        save_npz(
            pending,
            ids=dev["ids"][di],
            logits=pred,
            per_head=individual,
            threshold=np.array(cutoff),
            cal_ids=raw["ids"][cal],
            cal_y=raw["labels"][cal],
            cal_logits=calibration,
            **ablated,
        )
        temp = out / f"seed_{seed}.part"
        torch.save({"layer": layer.state_dict(), "head": head.state_dict(), "config_sha256": config_hash}, temp)
        temp.replace(out / f"seed_{seed}.pt")
        restored = torch.load(out / f"seed_{seed}.pt", map_location="cuda", weights_only=True)
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, restored["head"][key], rtol=0, atol=0)
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, restored["layer"][key], rtol=0, atol=0)

        record = {
            "seed": seed,
            "metrics": result["mean"],
            "ablations": branch,
            "traces": traces,
            "train_seconds": train_seconds,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "evaluation_seconds": time.perf_counter() - eval_started,
            "checkpoint_reload_verified": True,
            "trainable_params": nparams,
            "config_sha256": config_hash,
            "library_sha256": library_hash,
        }
        write_json(out / f"seed_{seed}.json", record)
        scores.append(pred)
        thresholds.append(cutoff)
        head_values.append(individual)
        seed_values.append(record)
        print(
            f"[{cfg['name']}] seed={seed} AP={result['mean']['ap']:.5f} F1={result['mean']['bot_f1']:.5f} "
            f"Churn={result['mean']['churn']*100:.3f}% (t={train_seconds:.1f}s)",
            flush=True,
        )

    save_npz(
        out / "predictions.npz",
        ids=dev["ids"][di],
        labels=dev["labels"][di],
        seeds=np.array(cfg["seeds"]),
        logits=np.stack(scores, axis=1),
        per_head_logits=np.stack(head_values, axis=1),
        thresholds=np.array(thresholds),
        library_sha256=np.array(library_hash),
        method=np.array(cfg["name"]),
        view_slots=bank,
    )
    result, _ = evaluate(np.stack(scores, axis=1), dev["labels"][di], np.array(thresholds))
    write_json(out / "summary.json", {
        "stage": stage,
        "config_sha256": config_hash,
        "results": result,
        "runtime": seed_values,
        "test_scored": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    args = parser.parse_args()
    run_single_config(args.config, args.stage)


if __name__ == "__main__":
    main()
