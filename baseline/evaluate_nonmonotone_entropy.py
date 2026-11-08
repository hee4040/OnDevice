#!/usr/bin/env python3
"""Exploratory five-fold CV on the original calibration half, with frozen complexity."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_adaptive_drop import (validate_cache, arrays, select_policy, apply_policy,
    fixed_codes, outcomes, WIDE_COSTS, NARROW_MACS, save_json, save_csv)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('fixed50', 'monotone_original', 'monotone_10bin', 'free_10bin')
LABELS = {'fixed50': 'Fixed 50%', 'monotone_original': 'Original monotone',
          'monotone_10bin': '10-bin monotone', 'free_10bin': '10-bin unrestricted'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slice_data(data, indices):
    return {key: value[indices] for key, value in data.items()}


def fit_binned(data, bins=10, monotone=False):
    routed = data['entropy'] >= 1
    e = data['entropy'][routed]
    if not len(e):
        return {'kind': 'binned', 'cuts': [], 'codes': [1], 'monotone': monotone,
                'candidate_count': 1, 'train_total_macs': int(NARROW_MACS) * len(routed),
                'train_budget_macs': int(NARROW_MACS) * len(routed)}
    cuts = np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1])
    bin_ids = np.searchsorted(cuts, e, side='right')
    n = np.bincount(bin_ids, minlength=bins).astype(np.int64)
    correct = np.stack([data['correct'][routed][bin_ids == b].sum(0) for b in range(bins)])
    assignments = np.array(list(itertools.product(range(3), repeat=bins)), dtype=np.int8)
    if monotone:
        assignments = assignments[(np.diff(assignments, axis=1) >= 0).all(1)]
    # Enumerate every bounded-complexity policy, including constant fixed50.
    costs = (WIDE_COSTS[assignments] * n).sum(1) + int(NARROW_MACS) * len(routed)
    scores = correct[np.arange(bins), assignments].sum(1) + data['narrow'][~routed].sum()
    budget = outcomes(data, fixed_codes(data))[1]
    feasible = np.flatnonzero(costs <= budget)
    # Product order is lexicographic; retain it as the last deterministic tie-break.
    best = feasible[np.lexsort((feasible, costs[feasible], -scores[feasible]))[0]]
    return {'kind': 'binned', 'monotone': monotone, 'cuts': cuts.tolist(),
            'codes': assignments[best].tolist(), 'train_bin_counts': n.tolist(),
            'train_correct': int(scores[best]), 'train_total_macs': int(costs[best]),
            'train_budget_macs': budget, 'candidate_count': len(assignments),
            'feasible_candidates': len(feasible),
            'tie_break': 'correct desc; integer MACs asc; bin-code vector lexicographic'}


def predict_binned(entropy, policy):
    bins = np.searchsorted(np.array(policy['cuts']), entropy, side='right')
    return np.where(entropy >= 1, np.array(policy['codes'], dtype=np.int64)[bins], -1)


def stratified_paired_ci(a, b, fold_ids, seed=0, repeats=5000):
    delta = a.astype(np.int8) - b.astype(np.int8)
    rng = np.random.default_rng(seed)
    sums = np.zeros(repeats, dtype=np.int64)
    for fold in np.unique(fold_ids):
        d = delta[fold_ids == fold]
        counts = np.bincount(d + 1, minlength=3)
        draws = rng.multinomial(len(d), counts / len(d), size=repeats)
        sums += draws[:, 2] - draws[:, 0]
    return np.quantile(100 * sums / len(delta), [.025, .975]).tolist()


def comparison(data, codes, name, fold):
    correct, cost = outcomes(data, codes)
    baseline, budget = outcomes(data, fixed_codes(data))
    return {'fold': fold, 'method': name, 'N': len(codes), 'routed_N': int((codes >= 0).sum()),
            'correct': int(correct.sum()), 'top1': 100 * float(correct.mean()),
            'delta_fixed50_pp': 100 * float((correct.astype(np.int8) - baseline.astype(np.int8)).mean()),
            'total_macs': cost, 'budget_macs': budget, 'gmacs_per_image': cost / len(codes) / 1e9,
            'budget_gmacs_per_image': budget / len(codes) / 1e9,
            'budget_exceeded': bool(cost > budget), 'budget_delta_percent': 100 * (cost / budget - 1),
            'exit_count': int((codes == -1).sum()), 'ratio25': int((codes == 0).sum()),
            'ratio50': int((codes == 1).sum()), 'ratio75': int((codes == 2).sum())}


def plot(out, policies, fold_rows):
    matrix = np.array([p['free_10bin']['codes'] for p in policies])
    fig, ax = plt.subplots(figsize=(10, 3.6), layout='constrained')
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(['#c7dcef', '#619ace', '#164e7c']), vmin=0, vmax=2, aspect='auto')
    ax.set_xticks(range(10), [f'B{i}' for i in range(1, 11)])
    ax.set_yticks(range(5), [f'Fold {i}' for i in range(1, 6)])
    for (y, x), code in np.ndenumerate(matrix):
        ax.text(x, y, f'{[25,50,75][code]}%', ha='center', va='center', color='white' if code == 2 else 'black')
    ax.set(xlabel='Entropy decile learned on each training fold (low → high)',
           title='Unrestricted policies selected on training folds')
    fig.savefig(out / 'fold_policies.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for name in METHODS[1:]:
        rows = [r for r in fold_rows if r['method'] == name]
        for ax, key in zip(axes, ['delta_fixed50_pp', 'budget_delta_percent']):
            ax.plot([r['fold'] for r in rows], [r[key] for r in rows], 'o-', label=LABELS[name])
    for ax in axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=1); ax.grid(alpha=.2)
        ax.set_xticks(range(1, 6)); ax.set_xlabel('Evaluation fold')
    axes[0].set_ylabel('Accuracy difference vs fixed50 (pp)')
    axes[1].set_ylabel('Cost difference vs fixed50 (%)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Out-of-fold observations · original calibration set only')
    fig.savefig(out / 'fold_comparison.png', dpi=180); plt.close(fig)


def report(out, summary, fold_rows):
    text = ['# 검증 2 · 비단조 entropy 선택기', '',
            '[원본] 기존 보정용 25,000장만 사용해 5-fold 교차검증을 수행했다. 각 fold의 20,000장으로 '
            '구간 경계와 정책을 선택하고 5,000장을 평가했다. 기존 holdout은 이번 집계에 사용하지 않았다. '
            'GPU 추론·백본 학습·새 특징 추출은 수행하지 않았다.', '',
            '[추론·작성자] 이전 손익 분석에서 이 보정용 전체의 집계 결과를 이미 확인했다. 따라서 이번 결과는 '
            '후속 설계를 위한 탐색이며, 새로운 독립 검증 성능이 아니다.', '',
            '## 비교 방법', '',
            '| 방법 | 정책 | 학습 시 후보 수 |', '|---|---|---:|',
            '| fixed50 | 항상 50% 유지 | 1 |',
            '| monotone_original | 기존 1% 분위수 경계쌍 탐색, 25→50→75 순서 | fold별 기록 |',
            '| monotone_10bin | 동일한 10개 entropy 구간, 유지율 감소 불가 | 66 |',
            '| free_10bin | 같은 10개 구간, 각 구간에서 25·50·75% 자유 선택 | 59,049 |', '',
            '구간 수 10, fold 수 5, seed 20261108을 집계 전에 고정했다. 추가 하이퍼파라미터 탐색은 하지 않았다. '
            '각 학습 fold의 고정 50% 비용 이하에서 정답 수 최대·비용 최소·코드 순서로 선택했다. '
            '비단조 방식도 단조 해와 고정 50%를 포함한다. 평가 fold의 비용이 초과해도 정책을 바꾸지 않았다.', '',
            '## Out-of-fold 결과 (25,000장)', '',
            '| 방법 | Top-1 (%) | 고정 50% 대비 (%p) | GMAC/장 | 예산 초과 fold | 전체 예산 초과 |',
            '|---|---:|---:|---:|---:|---|']
    for r in summary['pooled']:
        text.append(f"| {r['method']} | {r['top1']:.3f} | {r['delta_fixed50_pp']:+.3f} | {r['gmacs_per_image']:.6f} | {r['budget_exceeded_folds']}/5 | {r['budget_exceeded']} |")
    text += ['', '## 대응 비교와 불확실성', '',
             '| 비교 (앞−뒤) | 추가 정답 | 추가 오답 | 차이 (%p) | 95% 구간 (%p) |', '|---|---:|---:|---:|---|']
    for r in summary['paired']:
        text.append(f"| {r['a']} − {r['b']} | {r['improved']} | {r['regressed']} | {r['delta_pp']:+.3f} | [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] |")
    text += ['', '구간은 fold별 표본 수를 유지한 paired bootstrap 5,000회(seed 0)다. '
             '이미 학습된 정책을 고정한 예측 차이의 구간이며, fold 간 학습 자료 중복·정책 선택 변동·앞선 탐색을 반영한 확증적 신뢰구간이 아니다. '
             '예산 초과 fold가 있으면 모든 fold에서 예산을 지켰다고 주장하지 않는다.', '',
             '## Fold별 평가', '',
             '| fold | 방법 | Top-1 (%) | 차이 (%p) | 비용 차이 (%) |', '|---|---|---:|---:|---:|']
    for r in fold_rows:
        text.append(f"| {r['fold']} | {r['method']} | {r['top1']:.3f} | {r['delta_fixed50_pp']:+.3f} | {r['budget_delta_percent']:+.4f} |")
    text += ['', '![학습 fold에서 선택된 비단조 정책](fold_policies.png)', '',
             '![평가 fold의 정확도와 예산 변화](fold_comparison.png)', '',
             '## 저장 자료', '',
             'protocol.json: 실행 전 설정. folds.json: ID별 분할. policies/: 평가 전에 저장한 fold별 정책. '
             'predictions.csv: 25,000장의 out-of-fold 선택·정오. fold_comparison.csv와 pooled_comparison.csv: 정확도·정수 MAC 예산. '
             'paired_comparison.csv: 직접 비교. manifest.json: 입력·실행 소스 해시.', '',
             '```bash', summary['command'], '```', '']
    (out / 'report.md').write_text('\n'.join(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(); source = args.source.resolve()
    out = (args.output or ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ-nonmonotone-cv')).resolve()
    out.mkdir(parents=True, exist_ok=False); (out / 'policies').mkdir()
    protocol = {'scope': 'original_calibration_only_exploratory', 'folds': 5, 'fold_seed': 20261108,
                'fold_assignment': 'sorted calibration IDs; numpy default_rng permutation; five equal chunks',
                'bins': 10, 'bootstrap_repeats': 5000, 'bootstrap_seed': 0, 'methods': list(METHODS),
                'fit_budget': 'actual fixed50 integer MACs on each training fold',
                'evaluation_budget': 'actual fixed50 integer MACs on each evaluation fold; no retuning',
                'complexity_selection': 'fixed 10 bins; no hyperparameter search',
                'gate': 'student entropy >= 1.0', 'ratios': [.25, .5, .75],
                'interpretation': 'Exploratory, not a fresh independent holdout or formal generalization claim.'}
    save_json(out / 'protocol.json', protocol)
    expected = json.loads((ROOT / 'configs/adaptive_drop_20261107.json').read_text())
    meta = json.loads((source / 'cache/cache_metadata.json').read_text())
    if meta['weights_sha256'] != expected['expected_student_sha256'] or sha(source / 'cache/per_image.pt') != meta['cache_sha256']:
        raise ValueError('Student or cache checksum mismatch')
    split = json.loads((source / 'split_ids.json').read_text())
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(20261107)).tolist()
    if split['calibration'] != perm[:25000] or split['holdout'] != perm[25000:]:
        raise ValueError('Original split mismatch')
    cache = torch.load(source / 'cache/per_image.pt', map_location='cpu', weights_only=True); validate_cache(cache)
    indices = cache['image_id'].argsort()[sorted(split['calibration'])]
    data = arrays(cache, indices)
    fold_ids = np.zeros(len(indices), dtype=np.int64)
    for f, rows in enumerate(np.array_split(np.random.default_rng(protocol['fold_seed']).permutation(len(indices)), 5)):
        fold_ids[rows] = f
    save_json(out / 'folds.json', {'seed': protocol['fold_seed'], 'folds': [data['ids'][fold_ids == f].tolist() for f in range(5)]})
    codes = {name: np.full(len(indices), -9, dtype=np.int64) for name in METHODS}
    fold_rows, all_policies = [], []
    for f in range(5):
        train = slice_data(data, fold_ids != f)
        policies = {'fixed50': {'kind': 'fixed', 'ratio': .5}, 'monotone_original': select_policy(train),
                    'monotone_10bin': fit_binned(train, 10, True), 'free_10bin': fit_binned(train, 10, False)}
        if policies['free_10bin']['train_correct'] < policies['monotone_10bin']['train_correct']:
            raise AssertionError('Unrestricted training optimum must dominate its monotone subset')
        save_json(out / 'policies' / f'fold_{f + 1}.json', policies)  # locked before evaluation
        all_policies.append(policies)
        test = slice_data(data, fold_ids == f)
        selected = {'fixed50': fixed_codes(test), 'monotone_original': apply_policy(test, policies['monotone_original']),
                    **{name: predict_binned(test['entropy'], policies[name]) for name in ('monotone_10bin', 'free_10bin')}}
        for name, choice in selected.items():
            codes[name][fold_ids == f] = choice
            fold_rows.append(comparison(test, choice, name, f + 1))
        print(f'Completed fold {f + 1}/5', flush=True)
    pooled, correct = [], {}
    for name in METHODS:
        if (codes[name] == -9).any(): raise AssertionError('Incomplete out-of-fold predictions')
        r = comparison(data, codes[name], name, 'pooled')
        r['budget_exceeded_folds'] = sum(x['budget_exceeded'] for x in fold_rows if x['method'] == name)
        pooled.append(r); correct[name] = outcomes(data, codes[name])[0]
    pairs = [('monotone_original', 'fixed50'), ('monotone_10bin', 'fixed50'),
             ('free_10bin', 'fixed50'), ('free_10bin', 'monotone_original'), ('free_10bin', 'monotone_10bin')]
    paired = []
    for a, b in pairs:
        lo, hi = stratified_paired_ci(correct[a], correct[b], fold_ids)
        paired.append({'a': a, 'b': b, 'improved': int((correct[a] & ~correct[b]).sum()),
                       'regressed': int((~correct[a] & correct[b]).sum()),
                       'delta_pp': 100 * float((correct[a].astype(np.int8) - correct[b].astype(np.int8)).mean()),
                       'ci_low_pp': lo, 'ci_high_pp': hi})
    save_csv(out / 'fold_comparison.csv', fold_rows); save_csv(out / 'pooled_comparison.csv', pooled)
    save_csv(out / 'paired_comparison.csv', paired)
    save_csv(out / 'predictions.csv', [{'image_id': int(data['ids'][i]), 'fold': int(fold_ids[i]) + 1,
        'entropy': float(data['entropy'][i]), **{f'{name}_code': int(codes[name][i]) for name in METHODS},
        **{f'{name}_correct': int(correct[name][i]) for name in METHODS}} for i in range(len(indices))])
    summary = {'scope': protocol['scope'], 'pooled': pooled, 'paired': paired,
               'command': f'{sys.executable} baseline/evaluate_nonmonotone_entropy.py --source {source}'}
    save_json(out / 'summary.json', summary); plot(out, all_policies, fold_rows); report(out, summary, fold_rows)
    save_json(out / 'manifest.json', {'status': 'complete', 'protocol': protocol, 'source': str(source),
        'cache_sha256': meta['cache_sha256'], 'student_sha256': meta['weights_sha256'],
        'split_sha256': sha(source / 'split_ids.json'), 'script_sha256': sha(Path(__file__)),
        'analysis_dependency_sha256': sha(ROOT / 'baseline/analyze_adaptive_drop.py'),
        'numpy': np.__version__, 'torch': str(torch.__version__), 'gpu_used': False,
        'finished_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summary, indent=2)); print(f'Results: {out}')


if __name__ == '__main__':
    main()
