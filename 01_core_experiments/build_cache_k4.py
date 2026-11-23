"""Cache the inputs to the last K=4 MiniLM transformer blocks (layer[-4], index 8 of 12) for train and dev only.

NEVER opens or touches Twibot-20/test.json (test_opened: False).
Also verifies fp16 cached activations forwarded through layer[-4:] match reference pooled embeddings (cosine > 0.9999).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from audit_twibot20_holdout import DEFAULT, account_id, items, write
from encode_own_tweets import Encoder
from prepare_twibot20_local import AUDIT, OUTPUT as PREVIOUS, SHARD, SPLITS, choose_tweets


def save_array(path, **values):
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as f:
        np.savez(f, **values)
    tmp.replace(path)


@torch.inference_mode()
def shard(encoder, rows, split, index, limit, output, k_layers=4):
    name = f"{split}_{index:05d}"
    metadata = output / f"{name}.npz"
    tensor = output / f"{name}.npy"
    account_ids = np.array([account_id(row["ID"]) for row in rows], dtype=np.int64)
    if metadata.exists() and tensor.exists():
        with np.load(metadata) as saved:
            np.testing.assert_array_equal(saved["ids"], account_ids)
            assert np.load(tensor, mmap_mode="r").shape == (saved["lengths"].sum(), 384)
            return float(saved["min_cosine"]), float(saved["min_replay_cosine"])

    groups = [choose_tweets(encoder, row, limit) for row in rows]
    texts = [text for group in groups for text in group]
    toks = encoder.tokenizer(texts, padding=False, truncation=True, max_length=128)
    lengths = np.array([len(line) for line in toks["input_ids"]], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    hidden = np.empty((offsets[-1], encoder.dimension), dtype=np.float16)
    pooled = np.empty((len(texts), encoder.dimension), dtype=np.float32)
    replay_pooled = np.empty((len(texts), encoder.dimension), dtype=np.float32)
    captured = []

    def capture_input(_module, args):
        captured.append(args[0])

    target_block = encoder.model.encoder.layer[-k_layers]
    tail_blocks = encoder.model.encoder.layer[-k_layers:]
    hook = target_block.register_forward_pre_hook(capture_input)
    try:
        order = lengths.argsort(kind="stable")
        for pos in range(0, len(texts), encoder.batch_size):
            idx = order[pos:pos + encoder.batch_size]
            padded = encoder.tokenizer.pad(
                [{key: toks[key][i] for key in toks} for i in idx],
                return_tensors="pt",
            ).to("cuda")
            result = encoder.model(**padded).last_hidden_state.float()
            mask = padded["attention_mask"].unsqueeze(-1)
            pooled[idx] = F.normalize((result * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1).cpu().numpy()

            assert len(captured) == 1
            inp_tensor = captured.pop()  # [B, L, 384]
            # Verify exact replay from fp16 cached hidden states through tail_blocks
            inp_fp16 = inp_tensor.half()
            mask_2d = padded["attention_mask"].bool()
            pad = torch.zeros_like(mask_2d, dtype=inp_fp16.dtype).masked_fill_(
                ~mask_2d, torch.finfo(inp_fp16.dtype).min
            )
            with torch.autocast("cuda", dtype=torch.float16):
                x = inp_fp16
                for blk in tail_blocks:
                    x = blk(x, attention_mask=pad[:, None, None])[0]
            captured.clear()
            replay_pooled[idx] = F.normalize(
                (x.float() * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1
            ).cpu().numpy()

            tokens = inp_fp16.cpu().numpy()
            for i, text_id in enumerate(idx):
                hidden[offsets[text_id]:offsets[text_id + 1]] = tokens[i, :lengths[text_id]]
    finally:
        hook.remove()

    counts = np.array([len(group) for group in groups], dtype=np.int32)
    with np.load(PREVIOUS / f"{name}.npz") as old:
        np.testing.assert_array_equal(old["ids"], account_ids)
        np.testing.assert_array_equal(old["count"], counts)
        reference = old["tweets"][np.arange(limit)[None] < counts[:, None]].astype(np.float32)

    norms = np.linalg.norm(reference, axis=-1) * np.linalg.norm(pooled, axis=-1)
    cosine = float(np.min((reference * pooled).sum(-1) / norms))
    r_norms = np.linalg.norm(reference, axis=-1) * np.linalg.norm(replay_pooled, axis=-1)
    replay_cosine = float(np.min((reference * replay_pooled).sum(-1) / r_norms))
    assert cosine > 0.9999 and replay_cosine > 0.9999 and np.isfinite(hidden).all(), (name, cosine, replay_cosine)

    tmp = tensor.with_suffix(".part")
    with tmp.open("wb") as f:
        np.save(f, hidden)
    tmp.replace(tensor)
    save_array(
        metadata,
        ids=account_ids,
        counts=counts,
        lengths=lengths,
        offsets=offsets,
        min_cosine=cosine,
        min_replay_cosine=replay_cosine,
    )
    return cosine, replay_cosine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "cache_k4")
    parser.add_argument("--k-layers", type=int, default=4)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available() and not __import__("site").ENABLE_USER_SITE
    sys.path.append(str(AUDIT / "encoder_deps"))

    protocol = json.loads((PREVIOUS / "features_protocol.json").read_text())
    for name, key in (("model.safetensors", "encoder_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        with (AUDIT / "encoder" / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == protocol[key]

    encoder = Encoder(AUDIT / "encoder", 192, 128, "cuda")
    assert len(encoder.model.encoder.layer) == 12
    assert set(SPLITS.keys()) == {"train", "dev"}

    run_config = {
        "input": protocol,
        "hook_layer_index": 12 - args.k_layers,
        "k_layers": args.k_layers,
        "gpu": torch.cuda.get_device_name(),
        "test_opened": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if (output / "cache_protocol.json").exists():
        assert json.loads((output / "cache_protocol.json").read_text(encoding="utf8")) == run_config
    else:
        write(output, "cache_protocol.json", run_config)

    print(f"CUDA {torch.cuda.get_device_name()} start K={args.k_layers} cache -> {output}", flush=True)
    split_summary = {}
    with ZipFile(DEFAULT / "Twibot-20.zip") as archive:
        for split, (n, limit) in SPLITS.items():
            assert split in ("train", "dev")
            digest = hashlib.sha256()
            cosines, r_cosines, rows, seen = [], [], [], 0
            with archive.open(f"Twibot-20/{split}.json") as stream:
                for row in items(stream, digest):
                    rows.append(row)
                    if len(rows) == SHARD:
                        c, rc = shard(encoder, rows, split, seen // SHARD, limit, output, args.k_layers)
                        cosines.append(c)
                        r_cosines.append(rc)
                        seen += len(rows)
                        rows.clear()
                        if seen % (10 * SHARD) == 0:
                            print(f"{split} {seen}/{n} (min_replay_cos={min(r_cosines):.6f})", flush=True)
            if rows:
                c, rc = shard(encoder, rows, split, seen // SHARD, limit, output, args.k_layers)
                cosines.append(c)
                r_cosines.append(rc)
                seen += len(rows)
            assert seen == n and digest.hexdigest() == protocol["source"][split]
            split_summary[split] = {
                "accounts": seen,
                "shards": len(cosines),
                "lowest_cosine_vs_original": min(cosines),
                "lowest_replay_cosine_vs_original": min(r_cosines),
            }
            print(split, split_summary[split], flush=True)

    write(output, "cache_complete.json", {
        "splits": split_summary,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test_opened": False,
    })


if __name__ == "__main__":
    main()
