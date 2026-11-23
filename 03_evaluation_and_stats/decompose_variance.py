"""Balanced crossed random-effects ANOVA, with interaction including residual noise."""
import argparse
from pathlib import Path

import numpy as np

from experiment_common import prediction_data, probability, save_npz, write_json


def components(z):
    z = np.asarray(z, dtype=np.float64)
    n, r, m = z.shape
    if r < 2 or m < 2:
        raise ValueError('Variance decomposition requires at least two training seeds and two views')
    grand = z.mean(axis=(1, 2), keepdims=True)
    seed = z.mean(axis=2, keepdims=True)
    view = z.mean(axis=1, keepdims=True)
    interaction = z - seed - view + grand
    ms_interaction = (interaction ** 2).sum(axis=(1, 2)) / ((r - 1) * (m - 1))
    ms_seed = m * ((seed - grand) ** 2).sum(axis=(1, 2)) / (r - 1)
    ms_view = r * ((view - grand) ** 2).sum(axis=(1, 2)) / (m - 1)
    return np.column_stack(((ms_seed - ms_interaction) / m,
                            (ms_view - ms_interaction) / r, ms_interaction))


def summarize(z):
    estimate = components(z)
    mean = estimate.mean(axis=0)
    total = mean.sum()
    return {'component_names': ['training', 'tweet_sampling', 'interaction_and_residual'],
            'mean_signed_estimates': mean.tolist(),
            'negative_account_counts': (estimate < 0).sum(0).tolist(),
            'nonnegative_display_only': np.maximum(mean, 0).tolist(),
            'signed_fractions': (mean / total).tolist() if total > 0 else None,
            'assumptions': 'balanced crossed independent seed/view effects; zero-mean additive random effects; one measurement per cell so interaction and residual cannot be separated; negative method-of-moments estimates retained'}, estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = prediction_data(args.predictions)
    logit, x = summarize(data['logits'].astype('d'))
    prob, y = summarize(probability(data['logits']))
    write_json(args.output / 'variance_components.json', {'logit': logit, 'probability': prob,
               'seed_count': len(data['seeds']), 'views': data['logits'].shape[2]})
    save_npz(args.output / 'account_variance_components.npz', ids=data['ids'], logit=x, probability=y)
    print(logit)


if __name__ == '__main__':
    main()
