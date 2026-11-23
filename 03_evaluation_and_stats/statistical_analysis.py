"""Paired account-cluster bootstrap with Holm-adjusted centered bootstrap tests."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from decompose_variance import components
from evaluate_stability import account_metrics
from experiment_common import prediction_data, probability, save_npz, write_json


def estimates(d):
    logits, y, thresholds = d['logits'], d['labels'], d['thresholds']
    metrics = account_metrics(logits, y, thresholds)
    p = probability(logits)
    ap = np.mean([average_precision_score(y, p[:, r, m]) for r in range(p.shape[1]) for m in range(p.shape[2])])
    values = [ap, metrics['churn'].mean(), metrics['stable_error'][y == 1].mean(),
              metrics['stable_error'][y == 0].mean(), metrics['logit_variance'].mean()]
    values.extend(components(logits).mean(0)[:2] if logits.shape[1] >= 2 else [np.nan, np.nan])
    return np.asarray(values, dtype=float)


def compare(reference, candidate, iterations, seed):
    for key in ('ids', 'labels', 'seeds', 'library_sha256', 'view_slots'):
        np.testing.assert_array_equal(reference[key], candidate[key])
    if reference['logits'].shape != candidate['logits'].shape:
        raise ValueError('Both models must use identical account/seed/view axes')
    labels = reference['labels']
    if len(np.unique(labels)) != 2:
        raise ValueError('Both classes required for AP and two stable-error rates')
    measured = estimates(candidate) - estimates(reference)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == c) for c in (0, 1)]
    samples = np.empty((iterations, len(measured)))
    for b in range(iterations):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        subset = lambda d: dict(d, logits=d['logits'][ids], labels=d['labels'][ids])
        samples[b] = estimates(subset(candidate)) - estimates(subset(reference))
    names = ['ap', 'churn', 'bot_stable_miss', 'human_stable_false_positive', 'within_account_logit_variance',
             'training_variance', 'sampling_variance']
    rows = {}
    for j, name in enumerate(names):
        if not np.isfinite(measured[j]):
            rows[name] = {'effect': None, 'ci95': None, 'p_two_sided': None}
            continue
        centered = samples[:, j] - measured[j]
        p = (1 + np.count_nonzero(np.abs(centered) >= abs(measured[j]))) / (iterations + 1)
        rows[name] = {'effect': float(measured[j]), 'ci95': np.quantile(samples[:, j], [.025, .975]).tolist(),
                      'p_two_sided': p}
    valid = sorted([n for n in names if rows[n]['p_two_sided'] is not None], key=lambda n: rows[n]['p_two_sided'])
    adjusted = 0.
    for rank, name in enumerate(valid):
        adjusted = max(adjusted, min(1., (len(valid) - rank) * rows[name]['p_two_sided']))
        rows[name]['p_holm'] = adjusted
    # AP noninferiority uses a prespecified margin and one-sided confidence bound.
    rows['ap']['noninferiority_margin'] = .001
    rows['ap']['one_sided_lower95'] = float(np.quantile(samples[:, 0], .05))
    rows['ap']['noninferior'] = rows['ap']['one_sided_lower95'] > -.001
    rows['ap']['p_noninferiority_unadjusted'] = float((1+np.count_nonzero(samples[:,0]-measured[0]>=measured[0]+.001))/(iterations+1))
    return rows, samples


def finalize_family(root):
    """Global zero-effect Holm family plus a separate AP noninferiority family."""
    root=Path(root);files=sorted(root.glob('*/paired_bootstrap.json'));tests=[];ap_tests=[]
    for path in files:
        document=json.loads(path.read_text(encoding='utf8'))
        with np.load(path.parent/'bootstrap_effects.npz') as saved:samples=saved['effects']
        ap=document['metrics']['ap'];effect=ap['effect']
        ap['p_noninferiority_unadjusted']=float((1+np.count_nonzero(samples[:,0]-effect>=effect+.001))/(len(samples)+1))
        ap['simultaneous_bonferroni_lower95']=float(np.quantile(samples[:,0],.05/len(files)))
        write_json(path,document)
        for name,row in document['metrics'].items():
            if row['p_two_sided'] is not None:tests.append({'comparison':path.parent.name,'metric':name,**row})
        ap_tests.append({'comparison':path.parent.name,**ap})
    def adjust(rows,pvalue,target):
        previous=0.
        for rank,row in enumerate(sorted(rows,key=lambda x:x[pvalue])):
            previous=max(previous,min(1.,(len(rows)-rank)*row[pvalue]));row[target]=previous
    adjust(tests,'p_two_sided','p_holm_all_comparisons')
    adjust(ap_tests,'p_noninferiority_unadjusted','p_noninferiority_holm')
    for row in ap_tests:row['ap_noninferior_family_adjusted']=row['p_noninferiority_holm']<.05
    write_json(root/'holm_all_comparisons.json',{'scope':'all zero-effect endpoints in all prespecified comparisons; AP noninferiority is a separate prespecified one-sided family',
        'tests':tests,'ap_noninferiority_tests':ap_tests,'pilot_caveat':'conditional on trained seeds and fixed view library; not final confirmation'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261118)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError('At least 1,000 account-cluster bootstrap resamples required')
    rows, samples = compare(prediction_data(args.reference), prediction_data(args.candidate), args.iterations, args.seed)
    write_json(args.output / 'paired_bootstrap.json', {'metrics': rows, 'iterations': args.iterations,
               'unit': 'account, stratified by label; all training seeds and views retained together',
               'scope': 'conditional on trained seeds and fixed view library; Holm family is seven endpoints in this comparison',
               'test': 'two-sided centered paired cluster bootstrap; percentile confidence intervals', 'seed': args.seed})
    save_npz(args.output / 'bootstrap_effects.npz', effects=samples)
    print(rows)


if __name__ == '__main__':
    main()
