#!/usr/bin/env python3
"""Colab-ready, paired accuracy and end-to-end batch-1 latency for layer-wise drop."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch

from evaluate_drop import ROOT, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from layerwise_drop import NARROW_MACS, experiment_configs, layerwise_logits, wide_macs
from selective_refinement import forward_narrow_with_attention, make_update_mask


def save_json(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(path):
    def run(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    return dict(commit=run('rev-parse', 'HEAD'), status=run('status', '--short'))


def summarize(data, configs, threshold):
    routed = data['entropy'] >= threshold
    narrow, full = data['narrow_correct'], data['full_correct']
    fraction = routed.double().mean().item()
    full_cost = wide_macs([197] * 12)
    order = torch.argsort(data['entropy'], descending=True)
    rows = []
    for config in configs:
        name = config['name']
        cost = wide_macs(config['counts'])
        correct = data[name + '_correct']
        mixed = torch.where(routed, correct, narrow)
        # Nearest reachable entropy-routing budget; ties can make this point unattainable.
        count = round(int(routed.sum()) * cost / full_cost)
        matched = torch.zeros_like(routed)
        matched[order[:count]] = True
        tie = 0 < count < len(order) and bool(data['entropy'][order[count - 1]] == data['entropy'][order[count]])
        reference = torch.where(matched, full, narrow)
        row = dict(**config, samples=len(narrow), routed=int(routed.sum()),
                   top1_all_6h=100 * correct.double().mean().item(),
                   top1_routed=100 * correct[routed].double().mean().item() if routed.any() else None,
                   top1=100 * mixed.double().mean().item(),
                   avg_gmacs=(NARROW_MACS + fraction * cost) / 1e9,
                   wide_gmacs=cost / 1e9,
                   threshold_budget_reference_top1=100 * reference.double().mean().item(),
                   threshold_budget_reference_gmacs=(NARROW_MACS + count / len(narrow) * full_cost) / 1e9,
                   threshold_budget_reference_boundary_tie=tie)
        control = config['matched_uniform']
        if control:
            other = next(c for c in configs if c['name'] == control)
            control_correct = torch.where(routed, data[control + '_correct'], narrow)
            diff = mixed.double() - control_correct.double()
            row.update(uniform_gain_pp=100 * diff.mean().item(),
                       # Paired normal interval, fixed prespecified comparison (no selection claim).
                       uniform_gain_ci95_pp=[100 * (diff.mean().item() + sign * 1.96 * diff.std().item() / math.sqrt(len(diff)))
                                             for sign in (-1, 1)] if len(diff) > 1 else None,
                       uniform_cost_delta_percent=100 * (cost / wide_macs(other['counts']) - 1))
        rows.append(row)
    return dict(threshold=threshold, narrow_top1=100 * narrow.double().mean().item(),
                compute='Analytic MACs; excludes sorting/gather, normalization, softmax, entropy and dispatch', rows=rows)


def pipeline(model, image, config, threshold):
    # Full baseline avoids capture/ranking overhead; drop includes both.
    if config['name'] == 'full':
        model.p = 0.5
        features = model.forward_features(image, 0.5)
        logits = model.forward_head(features)
        attention = None
    else:
        features, logits, attention = forward_narrow_with_attention(model, image)
    routed = model.entropy(logits).item() >= threshold
    if routed:
        x0 = wide_input(model, image, features)
        logits = wide_logits(model, x0) if attention is None else layerwise_logits(model, x0, attention, config['ratios'])
    return logits, routed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument('--output', type=Path, required=True, help='Persistent run directory; identical runs resume')
    parser.add_argument('--mode', choices=['accuracy', 'latency'], default='accuracy')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--threads', type=int, default=1, help='CPU threads')
    parser.add_argument('--configs', nargs='+', help='Names to measure (latency only)')
    args = parser.parse_args()
    if not 1 <= args.samples <= 50000 or args.batch_size < 1 or args.workers < 0 or not math.isfinite(args.threshold):
        parser.error('Invalid samples, batch size, workers or threshold')
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error('Invalid latency settings')
    if args.configs and args.mode != 'latency':
        parser.error('--configs is for latency only; accuracy evaluates all paired controls')
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = load_model(args.repo.resolve(), args.checkpoint.resolve(), device=args.device)
    if len(model.blocks) != 12 or model.embed_dim != 384:
        raise RuntimeError('Expected 12 blocks and width 384')
    from prepare_imagenet import check_counts
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    check_counts(args.data_dir / 'validation', 50000)
    config = resolve_data_config({}, model=model, use_test_size=True)
    dataset = ImageFolder(args.data_dir / 'validation', transform=create_transform(**config, is_training=False))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.samples].tolist()
    configs = experiment_configs()
    if args.configs:
        if set(args.configs) - {c['name'] for c in configs}:
            parser.error('Unknown configuration name; see layerwise_drop.experiment_configs()')
        configs = [c for c in configs if c['name'] in args.configs]
    source = [Path(__file__), ROOT / 'baseline/layerwise_drop.py', ROOT / 'baseline/selective_refinement.py',
              ROOT / 'baseline/evaluate_drop.py', ROOT / 'baseline/evaluate_h1_diagnosis.py']
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    project=git_state(ROOT), official=git_state(args.repo),
                    checkpoint_sha256=sha(args.checkpoint), sources={p.name: sha(p) for p in source},
                    torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
                    device=torch.cuda.get_device_name() if args.device == 'cuda' else platform.processor(),
                    precision='float32, TF32 allowed', preprocessing=str(dataset.transform),
                    dataset_source=json.loads((args.data_dir / '.preparation/source.json').read_text())
                    if (args.data_dir / '.preparation/source.json').exists() else None,
                    dataset_manifest_sha256=hashlib.sha256(json.dumps(dataset.samples).encode()).hexdigest(),
                    indices=indices, configs=configs)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'environment.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != metadata:
            raise RuntimeError('Run settings/source/environment changed. Use a new --output directory.')
    else:
        save_json(manifest, metadata)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size if args.mode == 'accuracy' else 1,
                        num_workers=args.workers, pin_memory=args.device == 'cuda')
    sync = torch.cuda.synchronize if args.device == 'cuda' else lambda: None
    warmed = set()
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader):
            file = args.output / f'batch_{batch:05d}.pt'
            if file.exists():
                continue
            images, targets = images.to(args.device), targets.to(args.device)
            if args.mode == 'latency':
                result = {}
                for c in configs:
                    for _ in range(args.warmup if c['name'] not in warmed else 0):
                        pipeline(model, images, c, args.threshold)
                    if c['name'] not in warmed:
                        for _ in range(args.warmup):
                            pipeline(model, images, c, -1.0)
                        warmed.add(c['name'])
                    times = []
                    for _ in range(args.repeats):
                        sync()
                        start = time.perf_counter()
                        _, routed = pipeline(model, images, c, args.threshold)
                        sync()
                        times.append((time.perf_counter() - start) * 1000)
                    result[c['name']] = dict(ms=times, routed=routed)
            else:
                features, narrow, attention = forward_narrow_with_attention(model, images)
                x0 = wide_input(model, images, features)
                full = wide_logits(model, x0)
                # Every new batch also verifies the actual short-sequence implementation.
                official = model.forward_head(model.forward_features(images, 1.0, features, (3, 6)))
                torch.testing.assert_close(full, official, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(layerwise_logits(model, x0, attention, [1.] * 12), full, rtol=1e-5, atol=1e-6)
                if batch == 0:
                    for ratio in (0.25, 0.5, 0.75):
                        old = wide_logits(model, x0, make_update_mask(attention, ratio))
                        new = layerwise_logits(model, x0, attention, [ratio] * 12)
                        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                result = dict(entropy=model.entropy(narrow)[:, 0].cpu(),
                              narrow_correct=(narrow.argmax(1) == targets).cpu(),
                              targets=targets.cpu(), indices=torch.tensor(indices[batch * args.batch_size:(batch + 1) * args.batch_size]))
                for c in configs:
                    logits = full if c['name'] == 'full' else layerwise_logits(model, x0, attention, c['ratios'])
                    result[c['name'] + '_correct'] = (logits.argmax(1) == targets).cpu()
            tmp = file.with_suffix('.partial')
            torch.save(result, tmp)
            tmp.replace(file)
            if batch % 10 == 0:
                print(f'{args.mode}: {batch + 1}/{len(loader)} batches saved', flush=True)
    chunks = [torch.load(args.output / f'batch_{i:05d}.pt', weights_only=True) for i in range(len(loader))]
    if args.mode == 'accuracy':
        data = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        if len(data['entropy']) != args.samples:
            raise RuntimeError('Incomplete evaluation')
        summary = summarize(data, configs, args.threshold)
    else:
        rows = []
        for c in configs:
            ms = torch.tensor([chunk[c['name']]['ms'] for chunk in chunks], dtype=torch.float64)
            # Average repetitions per image, then percentiles across images.
            means = ms.mean(1)
            rows.append(dict(name=c['name'], samples=len(chunks), mean_ms=means.mean().item(),
                             p50_ms=means.quantile(.5).item(), p95_ms=means.quantile(.95).item(),
                             routed_fraction=sum(chunk[c['name']]['routed'] for chunk in chunks) / len(chunks)))
        summary = dict(measurement='Batch-1 full inference: 3H, attention capture for drop, entropy gate, optional 6H, sorting/gather; excludes loading, preprocessing and H2D', rows=rows)
    save_json(args.output / 'summary.json', summary)
    rows = summary['rows']
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Completed: {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
