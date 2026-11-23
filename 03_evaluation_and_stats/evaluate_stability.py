"""Metrics for independently inferred eight-tweet views of the same accounts."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from experiment_common import prediction_data, probability, write_json


def account_metrics(logits, labels, thresholds):
    n, r, m = logits.shape
    if m < 2:
        raise ValueError('At least two views are required for churn')
    p = probability(logits)
    decision = p >= thresholds[None, :, None]
    positives = decision.sum(axis=2)
    churn = 2 * positives * (m - positives) / (m * (m - 1))
    error = decision != labels[:, None, None]
    return {'churn': churn, 'logit_variance': logits.var(axis=2, ddof=1),
            'probability_variance': p.var(axis=2, ddof=1),
            'stable_error': error.all(axis=2), 'mostly_error': error.mean(axis=2) >= .9,
            'mean_probability': p.mean(axis=2)}


def evaluate(logits, labels, thresholds):
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError('Labels must be 0=Human and 1=Bot')
    a = account_metrics(logits, labels, thresholds)
    p = probability(logits)
    decisions = p >= thresholds[None, :, None]
    per_seed = []
    for seed in range(logits.shape[1]):
        each = []
        for view in range(logits.shape[2]):
            d, score = decisions[:, seed, view], p[:, seed, view]
            both = len(np.unique(labels)) == 2
            each.append({'accuracy': accuracy_score(labels, d),
                         'precision': precision_score(labels, d, zero_division=0),
                         'recall': recall_score(labels, d, zero_division=0),
                         'bot_f1': f1_score(labels, d, zero_division=0),
                         'macro_f1': f1_score(labels, d, labels=[0, 1], average='macro', zero_division=0),
                         'roc_auc': roc_auc_score(labels, score) if both else None,
                         'ap': average_precision_score(labels, score) if both else None})
        row = {k: float(np.mean([v[k] for v in each])) if each[0][k] is not None else None for k in each[0]}
        for name, values in a.items():
            row[name] = float(values[:, seed].mean())
        row.update(churn_median=float(np.median(a['churn'][:, seed])),
                   churn_p90=float(np.quantile(a['churn'][:, seed], .9)),
                   churn_p95=float(np.quantile(a['churn'][:, seed], .95)))
        for value, cls in ((0, 'human'), (1, 'bot')):
            selected = labels == value
            for metric in ('churn', 'stable_error', 'mostly_error', 'logit_variance', 'probability_variance'):
                row[f'{cls}_{metric}'] = float(a[metric][selected, seed].mean()) if selected.any() else None
        per_seed.append(row)
    means = {k: float(np.mean([s[k] for s in per_seed])) if per_seed[0][k] is not None else None for k in per_seed[0]}
    return {'accounts': len(labels), 'views': logits.shape[2], 'seed_count': logits.shape[1],
            'mean': means, 'per_seed': per_seed}, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = prediction_data(args.predictions)
    result, values = evaluate(data['logits'], data['labels'], data['thresholds'])
    result.update(method=str(data.get('method', 'unknown')), library_sha256=str(data['library_sha256']))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'metrics.json', result)
    with (args.output / 'account_metrics.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['account_id', 'label', 'seed', *values])
        writer.writeheader()
        for i, identity in enumerate(data['ids']):
            for r, seed in enumerate(data['seeds']):
                writer.writerow(dict(account_id=int(identity), label=int(data['labels'][i]), seed=int(seed),
                                     **{k: float(v[i, r]) for k, v in values.items()}))
    print(result['mean'])


if __name__ == '__main__':
    main()
