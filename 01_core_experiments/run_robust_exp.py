"""Algorithmic exploration for fixed 8-tweet bot detection stability on top of EXP-8-MH5.

Implements clean single-factor variants:
1. subbag_probvar: Zero-encoder-cost S=8 sub-bag sampling from 16 cached tweets,
   smooth-worst-case CVaR BCE + boundary-focused probability variance + anti-collapse floor.
2. resgate_subbag: Decoupled static base + intra-bag tweet self-consistency residual gate
   + subbag_probvar training.
3. setattn_subbag: Per-head zero-initialized residual set attention with entropy regularization
   + subbag_probvar training.
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "scripts"), str(ROOT / "work")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiment_common import read_config, save_npz, sha, write_json
from evaluate_stability import evaluate
import train_twibot20_local as base
from encode_own_tweets import Encoder
from prepare_twibot20_local import AUDIT
from train_twibot20_resampling_factorial import IndexedCache, gather_views
from run_experiment import predict_bank, slots_for, stratified_limit


class RobustSharedHeads(nn.Module):
    def __init__(self, original, count, arch="standard", attn_temp=1.0, gate_init_tau=0.9):
        super().__init__()
        self.text = deepcopy(original.text)
        self.profile = deepcopy(original.profile)
        self.heads = nn.ModuleList([deepcopy(original.head)])
        for _ in range(count - 1):
            head = deepcopy(original.head)
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            self.heads.append(head)
        self.arch = arch
        self.attn_temp = attn_temp
        if arch == "setattn":
            # Per-head scalar scorer initialized at 0 -> exact uniform mean pooling at step 0.
            self.attn_scorers = nn.ModuleList([nn.Linear(64, 1) for _ in range(count)])
            for scorer in self.attn_scorers:
                nn.init.zeros_(scorer.weight)
                nn.init.zeros_(scorer.bias)
        elif arch == "resgate":
            # Smooth self-consistency gate parameters: gamma = sigmoid(exp(log_scale) * (tau - disp))
            self.gate_log_scale = nn.Parameter(torch.tensor(1.5))
            self.gate_tau = nn.Parameter(torch.tensor(float(gate_init_tau)))

    def pool_standard(self, x, mask):
        # x: [V, B, 8, 64], mask: [V, B, 8]
        m = mask.float()
        return (x * m[..., None]).sum(-2) / m.sum(-1).clamp_min(1.0)[..., None]

    def pool_setattn(self, x, mask):
        # Returns per-head pooled [H, V, B, 64] and KL(w || uniform)
        m = mask.bool()
        V, B, K, _ = x.shape
        denom = m.float().sum(-1, keepdim=True).clamp_min(1.0)
        unif = m.float() / denom  # [V, B, K]
        pooled_list = []
        kl_total = x.new_tensor(0.0)
        for scorer in self.attn_scorers:
            raw_a = scorer(x).squeeze(-1) / self.attn_temp  # [V, B, K]
            raw_a = raw_a.masked_fill(~m, -1e4)
            w = torch.softmax(raw_a, dim=-1) * m.float()
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
            pooled_list.append((x * w[..., None]).sum(-2))
            # KL(w || unif) over valid slots
            log_ratio = torch.where(m & (w > 1e-8), torch.log((w + 1e-8) / (unif + 1e-8)), 0.0)
            kl_total = kl_total + (w * log_ratio).sum(-1).mean()
        return torch.stack(pooled_list, dim=0), kl_total / len(self.attn_scorers)

    def pieces_from_tweets(self, x, mask, profile_tensor):
        # x: [V, B, 8, 64], mask: [V, B, 8], profile_tensor: [B, 392]
        p_feat = self.profile(profile_tensor)[None]  # [1, B, 64]
        if self.arch == "setattn":
            pooled_h, attn_kl = self.pool_setattn(x, mask)  # [H, V, B, 64]
            p_exp = p_feat.expand(x.shape[0], -1, -1)  # [V, B, 64]
            logits = torch.stack([
                h(torch.cat((pooled_h[i], p_exp), dim=-1)).squeeze(-1)
                for i, h in enumerate(self.heads)
            ])
            pooled_mean = self.pool_standard(x, mask)
            fused = torch.cat((pooled_mean, p_exp), dim=-1)
            return logits, pooled_mean, p_exp, fused, {"attn_kl": attn_kl}

        pooled = self.pool_standard(x, mask)  # [V, B, 64]
        p_exp = p_feat.expand_as(pooled)
        fused = torch.cat((pooled, p_exp), dim=-1)
        logits_full = torch.stack([h(fused).squeeze(-1) for h in self.heads])  # [H, V, B]

        if self.arch == "resgate":
            # Static base logits (zero tweet vector)
            fused_zero = torch.cat((torch.zeros_like(pooled), p_exp), dim=-1)
            logits_base = torch.stack([h(fused_zero).squeeze(-1) for h in self.heads])
            # Normalized RMS dispersion of the 8 tweets around their mean
            m = mask.float()
            diff = x - pooled.unsqueeze(-2)
            sq_dist = (diff * diff).sum(-1) * m
            rms_disp = torch.sqrt(sq_dist.sum(-1) / m.sum(-1).clamp_min(1.0) + 1e-6)
            mu_norm = torch.linalg.vector_norm(pooled, dim=-1).clamp_min(1e-2)
            norm_disp = rms_disp / mu_norm  # [V, B]
            scale = torch.exp(self.gate_log_scale.clamp(-2.0, 3.0))
            gamma = torch.sigmoid(scale * (self.gate_tau - norm_disp))  # [V, B]
            logits = logits_base + gamma.unsqueeze(0) * (logits_full - logits_base)
            return logits, pooled, p_exp, fused, {
                "logits_base": logits_base,
                "gamma": gamma,
                "norm_disp": norm_disp,
            }

        return logits_full, pooled, p_exp, fused, {}

    def pieces(self, view):
        x = self.text(view.tweets.float())
        logits, pooled, p_exp, fused, _ = self.pieces_from_tweets(x, view.mask, view.profile)
        return logits, pooled, p_exp, fused

    def forward(self, view):
        logits, *_ = self.pieces(view)
        if len(self.heads) == 1:
            return logits[0]
        return torch.logit(logits.sigmoid().mean(0).clamp(1e-7, 1 - 1e-7))


def sample_subbags_from_pool(counts, S, rng, budget=8, max_pool=16):
    """Return [S, B, budget] local indices into [0..min(c, max_pool)-1]."""
    B = len(counts)
    sub_slots = np.full((S, B, budget), -1, dtype=np.int64)
    for b, c in enumerate(counts):
        pool_n = min(int(c), max_pool)
        k = min(pool_n, budget)
        for s in range(S):
            sub_slots[s, b, :k] = rng.choice(pool_n, k, replace=False)
    return sub_slots


def subbag_objective(head, table_all, counts, profile_obj, sub_slots_np, cfg):
    """Zero-encoder-cost S-subset training loss using pre-encoded account tweet table."""
    # table_all: [B, 16, 384]
    x_all = head.text(table_all.float())  # [B, 16, 64]
    idx_t = torch.as_tensor(sub_slots_np, device=x_all.device, dtype=torch.long)  # [S, B, 8]
    mask = idx_t >= 0  # [S, B, 8]
    S, B, K = idx_t.shape
    b_idx = torch.arange(B, device=x_all.device)[None, :, None].expand(S, B, K)
    x_sub = x_all[b_idx, idx_t.clamp_min(0)].masked_fill(~mask[..., None], 0.0)  # [S, B, 8, 64]

    logits_hsb, pooled_sb, p_exp, _, extra = head.pieces_from_tweets(x_sub, mask, profile_obj.profile)
    # Ensemble combined logit per subset: [S, B]
    if len(head.heads) == 1:
        z_ens = logits_hsb[0]
    else:
        z_ens = torch.logit(logits_hsb.sigmoid().mean(0).clamp(1e-7, 1 - 1e-7))

    labels = profile_obj.labels  # [B]
    pos_weight = cfg["_pos_weight"]

    # 1. Per-head BCE across all S subsets, with optional smooth-worst-case CVaR emphasis
    y_hsb = labels[None, None, :].expand_as(logits_hsb)
    bce_hsb = F.binary_cross_entropy_with_logits(logits_hsb, y_hsb, pos_weight=pos_weight, reduction="none")  # [H, S, B]
    mean_bce = bce_hsb.mean()

    cvar_alpha = float(cfg.get("cvar_alpha", 0.0))
    if cvar_alpha > 0.0 and S > 1:
        # Smooth-max over subsets per head & account
        cvar_temp = float(cfg.get("cvar_temp", 1.0))
        bce_ens = F.binary_cross_entropy_with_logits(z_ens, labels[None, :].expand_as(z_ens), pos_weight=pos_weight, reduction="none")  # [S, B]
        smooth_max = cvar_temp * torch.logsumexp(bce_ens / cvar_temp, dim=0) - cvar_temp * np.log(S)
        cls_loss = (1.0 - cvar_alpha) * mean_bce + cvar_alpha * smooth_max.mean()
    else:
        cls_loss = mean_bce

    # 2. Boundary-focused probability-space variance across the S subsets
    t_var = float(cfg.get("var_temperature", 1.0))
    p_sub = torch.sigmoid(z_ens / t_var)  # [S, B]
    prob_var = (4.0 * t_var * t_var) * p_sub.var(dim=0, unbiased=False).mean()

    # Also optional pairwise KL across pairs if lambda_kl > 0
    loss = cls_loss + float(cfg.get("lambda_prob_var", 1.5)) * prob_var

    # 3. Anti-collapse constraint: keep tweet increment std above target floor
    lambda_ac = float(cfg.get("lambda_anticollapse", 0.05))
    if lambda_ac > 0.0:
        if "logits_base" in extra:
            lb = extra["logits_base"]
            z_stat = lb[0] if len(head.heads) == 1 else torch.logit(lb.sigmoid().mean(0).clamp(1e-7, 1 - 1e-7))
        else:
            fused_zero = torch.cat((torch.zeros_like(pooled_sb[:1]), p_exp[:1]), dim=-1)
            lb = torch.stack([h(fused_zero).squeeze(-1) for h in head.heads])
            z_stat = (lb[0] if len(head.heads) == 1 else torch.logit(lb.sigmoid().mean(0).clamp(1e-7, 1 - 1e-7))).squeeze(0)
        tweet_inc = z_ens - z_stat[None, :]  # [S, B]
        inc_std = tweet_inc.std()
        target_inc = float(cfg.get("target_inc_std", 1.70))
        ac_loss = F.relu(target_inc - inc_std) ** 2
        loss = loss + lambda_ac * ac_loss

    # 4. Arch-specific regularizers
    if "attn_kl" in extra:
        loss = loss + float(cfg.get("lambda_attn_ent", 0.05)) * extra["attn_kl"]
    if "gamma" in extra:
        # Keep mean gate active (prevent trivial gate->0 collapse)
        gate_floor = float(cfg.get("target_gate_mean", 0.70))
        loss = loss + float(cfg.get("lambda_gate_floor", 0.10)) * (F.relu(gate_floor - extra["gamma"].mean()) ** 2)
        # Mildly anchor static base to prevent upward drift on Humans
        lambda_stat_bce = float(cfg.get("lambda_static_bce", 0.05))
        if lambda_stat_bce > 0:
            lb = extra["logits_base"]  # [H, S, B] (constant over S)
            stat_bce = F.binary_cross_entropy_with_logits(lb[:, 0, :], labels[None, :].expand(lb.shape[0], B), pos_weight=pos_weight)
            loss = loss + lambda_stat_bce * stat_bce

    return loss, cls_loss, prob_var


def run_single_config(cfg_path, stage):
    cfg = read_config(cfg_path)
    cfg.update(cfg["stages"][stage])
    if cfg["unfreeze_layers"] not in (0, 1):
        raise ValueError("Only K=0 or 1 supported with existing prefix cache")
    if cfg["budget"] != 8:
        raise ValueError("Budget must be 8")

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
    torch.set_num_threads(cfg["cpu_threads"])
    template = deepcopy(encoder.model.encoder.layer[-1]).float().eval()
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
    caches = [IndexedCache(ROOT / cfg["prefix_cache"], split, d["ids"], d["count"], width)
              for split, d, width in (("train", raw, 16), ("dev", dev, 32))]
    cal_slots = slots_for(raw["count"][cal], 1, np.random.default_rng(cfg["sampling_seed"] + 3), 8)

    # Precompute full 16-slot pool matrix for fit accounts: [len(fit), 16]
    fit_counts = raw["count"][fit]
    pool_slots_fit = np.full((len(fit), 16), -1, dtype=np.int16)
    for i, c in enumerate(fit_counts):
        k = min(int(c), 16)
        pool_slots_fit[i, :k] = np.arange(k, dtype=np.int16)

    scores, thresholds, seed_values, head_values = [], [], [], []
    arch = cfg.get("arch", "standard")
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
        weights = torch.load(ROOT / cfg["initialization"] / f"seed_{seed}_tuned_natural.pt", map_location="cuda", weights_only=True)
        layer = deepcopy(template).requires_grad_(cfg["unfreeze_layers"] > 0).eval()
        layer.load_state_dict(weights["last_layer"])
        anchor = base.AccountHead().cuda()
        anchor.load_state_dict(weights["head"])
        head = RobustSharedHeads(
            anchor,
            cfg["heads"],
            arch=arch,
            attn_temp=float(cfg.get("attn_temp", 1.0)),
            gate_init_tau=float(cfg.get("gate_init_tau", 0.9)),
        ).cuda()

        nparams = sum(p.numel() for p in layer.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters())
        parameters = [p for p in list(layer.parameters()) + list(head.parameters()) if p.requires_grad]
        groups = [{"params": head.parameters(), "lr": cfg["head_lr"]}]
        if cfg["unfreeze_layers"]:
            groups.append({"params": layer.parameters(), "lr": cfg["encoder_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])

        rng = np.random.default_rng(cfg["sampling_seed"] + seed)
        pos = torch.tensor((raw["labels"][fit] == 0).sum() / (raw["labels"][fit] == 1).sum(), device="cuda", dtype=torch.float32)
        cfg["_pos_weight"] = pos

        traces = []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        for epoch in range(cfg["epochs"]):
            head.train()
            order = np.random.default_rng(seed + epoch * 1000).permutation(len(fit))
            losses = []
            for start in range(0, len(order), cfg["batch"]):
                idx = order[start:start + cfg["batch"]]
                table = caches[0].selected(layer, fit[idx], pool_slots_fit[idx, None, :])
                sub_slots = sample_subbags_from_pool(fit_counts[idx], S_bags, rng, budget=8, max_pool=16)
                loss, cls_val, pvar_val = subbag_objective(head, table, fit_counts[idx], profiles[0].at(idx), sub_slots, cfg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append((float(loss.detach()), float(cls_val.detach()), float(pvar_val.detach())))
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
            ids=dev["ids"][di], logits=pred, per_head=individual, threshold=np.array(cutoff),
            cal_ids=raw["ids"][cal], cal_y=raw["labels"][cal], cal_logits=calibration, **ablated,
        )
        temp = out / f"seed_{seed}.part"
        torch.save({"layer": layer.state_dict(), "head": head.state_dict(), "config_sha256": config_hash}, temp)
        temp.replace(out / f"seed_{seed}.pt")
        restored = torch.load(out / f"seed_{seed}.pt", map_location="cuda", weights_only=True)
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, restored["head"][key], rtol=0, atol=0)

        record = {
            "seed": seed, "metrics": result["mean"], "ablations": branch, "traces": traces,
            "train_seconds": train_seconds, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "evaluation_seconds": time.perf_counter() - eval_started,
            "checkpoint_reload_verified": True, "trainable_params": nparams,
            "config_sha256": config_hash, "library_sha256": library_hash,
        }
        write_json(out / f"seed_{seed}.json", record)
        scores.append(pred)
        thresholds.append(cutoff)
        head_values.append(individual)
        seed_values.append(record)
        print(f"[{cfg['name']}] seed={seed} AP={result['mean']['ap']:.5f} F1={result['mean']['bot_f1']:.5f} Churn={result['mean']['churn']*100:.3f}% (t={train_seconds:.1f}s)", flush=True)

    save_npz(
        out / "predictions.npz",
        ids=dev["ids"][di], labels=dev["labels"][di], seeds=np.array(cfg["seeds"]),
        logits=np.stack(scores, axis=1), per_head_logits=np.stack(head_values, axis=1),
        thresholds=np.array(thresholds), library_sha256=np.array(library_hash),
        method=np.array(cfg["name"]), view_slots=bank,
    )
    result, _ = evaluate(np.stack(scores, axis=1), dev["labels"][di], np.array(thresholds))
    write_json(out / "summary.json", {
        "stage": stage, "config_sha256": config_hash, "results": result,
        "runtime": seed_values, "test_scored": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="smoke")
    args = parser.parse_args()
    run_single_config(args.config, args.stage)


if __name__ == "__main__":
    main()
