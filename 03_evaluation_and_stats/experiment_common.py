"""Shared I/O and validation for the fixed-budget experimental system."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.part')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_config(path):
    # JSON syntax is an unambiguous YAML 1.2 subset and requires no YAML runtime.
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf8'))
    parent = raw.pop('extends', None)
    if parent:
        original = read_config(path.parent / parent)
        original.update(raw)
        raw = original
    return raw


def probability(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


def prediction_data(path):
    with np.load(path, allow_pickle=False) as saved:
        data = dict(saved)
    z = data['logits']
    if z.ndim != 3 or z.shape[0] != len(data['ids']):
        raise ValueError('logits must have shape [accounts, training seeds, views]')
    if len(np.unique(data['ids'])) != len(data['ids']) or not np.isfinite(z).all():
        raise ValueError('Duplicate account IDs or nonfinite predictions')
    if z.shape[1] != len(data['seeds']) or data['thresholds'].shape != (z.shape[1],):
        raise ValueError('Training seeds and thresholds do not match prediction axes')
    return data
