#!/usr/bin/env python3
"""Reassess saved predictions with ImageNet ReaL and sample regressions for inspection.

No model inference. ReaL entries are indexed by ORIGINAL validation filename, not
ImageFolder index. Empty annotation lists are excluded from ReaL metrics.
"""
import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_id(source_path):
    match = re.fullmatch(r'ILSVRC2012_val_(\d{8})_(n\d{8})\.JPEG', source_path)
    if not match or not 1 <= int(match[1]) <= 50000:
        raise ValueError(f'Invalid original validation filename: {source_path}')
    return int(match[1]), match[2]


def transition(narrow, wide):
    return {(True, True): 'both_correct', (False, True): 'corrected',
            (True, False): 'regressed', (False, False): 'both_wrong'}[bool(narrow), bool(wide)]


def assess(row, labels):
    if not labels:
        return {'real_group': 'excluded', 'real_narrow_correct': None,
                'real_wide_correct': None, 'real_routed_correct': None}
    n, w, r = (int(row[k]) in labels for k in ('narrow_prediction', 'wide_prediction', 'routed_prediction'))
    return {'real_group': transition(n, w), 'real_narrow_correct': int(n),
            'real_wide_correct': int(w), 'real_routed_correct': int(r)}


def metrics(rows):
    included = [r for r in rows if r['real_group'] != 'excluded']
    result = {'samples': len(rows), 'real_evaluated': len(included),
              'real_excluded': len(rows) - len(included),
              'original_groups': dict(Counter(r['group'] for r in rows)),
              'real_groups': dict(Counter(r['real_group'] for r in rows))}
    for stage in ('narrow', 'wide', 'routed'):
        result[stage] = {
            'original_correct_all': sum(int(r[f'{stage}_correct']) for r in rows),
            'original_top1_all': 100 * sum(int(r[f'{stage}_correct']) for r in rows) / len(rows) if rows else None,
            'original_correct_matched': sum(int(r[f'{stage}_correct']) for r in included),
            'original_top1_matched': 100 * sum(int(r[f'{stage}_correct']) for r in included) / len(included) if included else None,
            'real_correct': sum(r[f'real_{stage}_correct'] for r in included),
            'real_top1': 100 * sum(r[f'real_{stage}_correct'] for r in included) / len(included) if included else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--failures', type=Path, required=True)
    parser.add_argument('--real-labels', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--audit-samples', type=int, default=40)
    args = parser.parse_args()
    env = json.loads((args.source / 'environment.json').read_text())
    stage_summary = json.loads((args.source / 'summary.json').read_text())
    if not stage_summary.get('full_validation') or not stage_summary.get('complete'):
        parser.error('Full validation analysis required')
    failure_summary = json.loads((args.failures / 'summary.json').read_text())
    if (ROOT / failure_summary['source_run']).resolve() != args.source.resolve() or failure_summary['checkpoint_sha256'] != env['checkpoint_sha256']:
        raise ValueError('Failure analysis source/checkpoint mismatch')
    data = Path(env['arguments']['data_dir'])
    repo = Path(env['arguments']['repo'])
    synsets = (repo / 'timm/data/_info/imagenet_synsets.txt').read_text().splitlines()
    labels = json.loads(args.real_labels.read_text())
    if len(labels) != 50000 or any(not isinstance(x, list) or any(type(i) is not int or not 0 <= i < 1000 for i in x) for x in labels):
        raise ValueError('Invalid ReaL annotations')
    mapping = {}
    for manifest in sorted((data / '.preparation').glob('validation-?????-of-?????.json')):
        for image in json.loads(manifest.read_text())['images']:
            key = 'validation/' + image['path']
            if key in mapping:
                raise ValueError('Duplicate manifest path')
            mapping[key] = original_id(image['source_path'])
    rows = list(csv.DictReader((args.source / 'images.csv').open()))
    ids = []
    for row in rows:
        ident, synset = mapping[row['image_path']]
        if synsets[int(row['target'])] != synset:
            raise ValueError('Class index / original filename synset mismatch')
        ids.append(ident)
        accepted = labels[ident - 1]
        row.update(original_validation_id=ident, real_labels=json.dumps(accepted),
                   real_label_count=len(accepted), original_label_in_real=int(int(row['target']) in accepted))
        row.update(assess(row, accepted))
    if len(ids) != 50000 or set(ids) != set(range(1, 50001)):
        raise ValueError('Expected unique complete original validation IDs')
    routed = [r for r in rows if r['routed_to_6h'] == '1']
    regressions = [r for r in routed if r['group'] == 'regressed']
    details = {r['dataset_index']: r for r in csv.DictReader((args.failures / 'regressions_detailed.csv').open())}
    if set(details) != {r['dataset_index'] for r in regressions}:
        raise ValueError('Probability details do not match regression population')
    for r in regressions:
        d = details[r['dataset_index']]
        for key in ('target', 'narrow_prediction', 'wide_prediction', 'image_path'):
            if r[key] != d[key]:
                raise ValueError('Probability detail identity mismatch')
        if d['source_predictions_reproduced'] != '1':
            raise ValueError('Probability prediction did not reproduce')
    if not 1 <= args.audit_samples <= len(regressions):
        parser.error('Invalid sample size')
    selected = random.Random(args.seed).sample(sorted(regressions, key=lambda r: int(r['dataset_index'])), args.audit_samples)
    out = ROOT / 'results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-failure-audit')
    out.mkdir(parents=True)
    write_json(out / 'sample_indices.json', {'seed': args.seed, 'population': len(regressions),
               'method': 'random.Random.sample without replacement, population sorted by dataset_index',
               'dataset_indices': [int(r['dataset_index']) for r in selected]})
    write_csv(out / 'images_reassessed.csv', rows)
    enriched = [{**r, **{k: v for k, v in details[r['dataset_index']].items() if k not in r}} for r in regressions]
    write_csv(out / 'regressions_reassessed.csv', enriched)

    def probability_summary(subset):
        ds = [details[r['dataset_index']] for r in subset]
        return {'samples': len(ds),
            'true_probability_increased': sum(float(d['wide_true_probability']) > float(d['narrow_true_probability']) for d in ds),
            'wide_winner_was_narrow_runner_up': sum(json.loads(d['narrow_top5'])[1][0] == d['wide_prediction_name'] for d in ds),
            'wide_winner_was_narrow_top5': sum(d['wide_prediction_name'] in [p[0] for p in json.loads(d['narrow_top5'])] for d in ds),
            'entropy_decreased': sum(float(d['wide_entropy']) < float(d['narrow_entropy_rerun']) for d in ds),
            'true_rank_histogram': dict(Counter(d['wide_true_rank'] for d in ds)),
            'median_true_probability_change': statistics.median(float(d['wide_true_probability']) - float(d['narrow_true_probability']) for d in ds) if ds else None,
            'median_wrong_probability_change': statistics.median(float(d['wide_wrong_probability']) - float(d['narrow_wrong_probability']) for d in ds) if ds else None}

    bins = []
    for low, high in [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 7)]:
        subset = [r for r in rows if low <= float(r['narrow_entropy']) < high]
        bins.append({'lower_inclusive': low, 'upper_exclusive': high, **metrics(subset)})
    if sum(b['samples'] for b in bins) != len(rows):
        raise ValueError('Entropy bins do not cover population')
    summary = {'source': str(args.source), 'failures': str(args.failures),
        'real_provenance': json.loads(args.real_labels.with_name('source.json').read_text()),
        'real_sha256': hashlib.sha256(args.real_labels.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mapping_checks': '50000 unique original validation IDs, complete coverage, all target synsets match filenames',
        'all': metrics(rows), 'routed_to_6h': metrics(routed),
        'original_routed_regressions': metrics(regressions),
        'routed_transition_cross_table': [
            {'original_group': old, 'real_group': new, 'count': count}
            for (old, new), count in sorted(Counter((r['group'], r['real_group']) for r in routed).items())],
        'real_multiple_labels': sum(r['real_label_count'] > 1 for r in rows),
        'original_label_not_in_nonempty_real': sum(r['real_label_count'] > 0 and not r['original_label_in_real'] for r in rows),
        'regression_probabilities': probability_summary(regressions),
        'still_regressed_probabilities': probability_summary([r for r in regressions if r['real_group'] == 'regressed']),
        'entropy_bins': bins, 'audit_seed': args.seed, 'audit_samples': args.audit_samples,
        'limitations': 'ReaL is complementary reassessed annotation, not infallible truth. Empty lists are excluded, never counted wrong. '
        'Probabilities are from a regression-only rerun with all Top-1 predictions reproduced. '
        'Random visual inspection is non-expert and unblinded; it cannot establish causal token effects.'}
    if summary['real_sha256'] != summary['real_provenance']['sha256']:
        raise ValueError('Annotation checksum differs from recorded provenance')
    write_json(out / 'summary.json', summary)

    # Reuse the exact saved evaluation preprocessing; no weights or GPU required.
    sys.path.insert(0, str(repo))
    from timm.data import create_transform
    import torch
    from PIL import Image
    import numpy as np
    transform = create_transform(**env['preprocessing'], is_training=False)
    cfg = env['preprocessing']
    mean, std = torch.tensor(cfg['mean'])[:, None, None], torch.tensor(cfg['std'])[:, None, None]
    (out / 'model_inputs').mkdir()
    cards, audit = [], []
    for n, r in enumerate(selected):
        d = details[r['dataset_index']]
        with Image.open(data / r['image_path']) as im:
            tensor = transform(im.convert('RGB'))
        pixels = ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        crop = out / 'model_inputs' / f'case_{n:02d}.png'
        Image.fromarray(pixels).save(crop)
        entry = {**r, 'case': n, 'target_name': d['target_name'], 'wide_prediction_name': d['wide_prediction_name'],
                 'inspection_image': str(crop.relative_to(out)), 'visual_observation': ''}
        audit.append(entry)
        e = html.escape
        original = base64.b64encode((data / r['image_path']).read_bytes()).decode()
        encoded = base64.b64encode(crop.read_bytes()).decode()
        cards.append(f'<article data-group="{r["real_group"]}"><h2>{n:02d}. {e(d["target_name"])} → {e(d["wide_prediction_name"])}</h2>'
            f'<p>ReaL: <b>{r["real_group"]}</b> · index {r["dataset_index"]} · original ID {r["original_validation_id"]} · labels {r["real_labels"]}</p>'
            f'<img src="data:image/jpeg;base64,{original}" alt="Original"><img src="data:image/png;base64,{encoded}" alt="Actual model input">'
            f'<p>P(target): {float(d["narrow_true_probability"]):.3f} → {float(d["wide_true_probability"]):.3f}; '
            f'P(6H winner): {float(d["narrow_wrong_probability"]):.3f} → {float(d["wide_wrong_probability"]):.3f}; '
            f'target rank: {d["wide_true_rank"]}</p><details><summary>Top-5</summary><pre>3H: {e(d["narrow_top5"])}\n6H: {e(d["wide_top5"])}</pre></details></article>')
    write_json(out / 'visual_audit.json', audit)
    (out / 'gallery.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Random regression audit</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}article{border-top:1px solid #aaa;padding:20px 0}img{width:46%;height:300px;object-fit:contain}pre{white-space:pre-wrap}h2{font-size:20px}</style>'
        f'<h1>Random audit: {len(selected)} / {len(regressions)} original regressions</h1><p>Seed {args.seed}. Original (left), exact model crop (right). '
        'ReaL groups compare 3H and 6H using reassessed labels; excluded means no accepted label. '
        'This is an unblinded non-expert visual sample, not a causal diagnosis.</p>'
        '<label>ReaL group: <select onchange="document.querySelectorAll(\'article\').forEach(a=>a.hidden=this.value!==\'all\'&&a.dataset.group!==this.value)">'
        + ''.join(f'<option>{g}</option>' for g in ['all', 'both_correct', 'regressed', 'corrected', 'both_wrong', 'excluded'])
        + '</select></label>' + ''.join(cards) + '</html>')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('entropy_bins',)}, indent=2))
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
