#!/usr/bin/env python3
"""Analyze saved 3H CLS attention; no model inference or GPU required."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def patch_metrics(attention):
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 197 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected finite nonnegative [N,197] attention')
    patches = a[:, 1:]
    mass = patches.sum(1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError('Patch attention mass must be positive')
    p = patches / mass
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    entropy = -(p * logp).sum(1)
    top10 = np.sort(p, axis=1)[:, -10:].sum(1)
    return p, entropy, top10


def load_source(source):
    summary = json.loads((source / 'summary.json').read_text())
    if not summary.get('complete') or not summary.get('full_validation'):
        raise ValueError('Completed full-validation source required')
    rows = list(csv.DictReader((source / 'images.csv').open()))
    a = np.load(source / 'narrow_cls_attention.npy', mmap_mode='r')
    if len(rows) != 50000 or a.shape != (50000, 197):
        raise ValueError('Incomplete source')
    if {int(r['dataset_index']) for r in rows} != set(range(50000)):
        raise ValueError('Duplicate/missing dataset indices')
    if [int(r['attention_row']) for r in rows] != list(range(50000)):
        raise ValueError('CSV/attention row alignment mismatch')
    if not np.allclose(a.sum(1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError('Source full attention does not sum to one')
    return rows, a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', type=Path, default=ROOT / 'baseline/reports/2026-10-15/illustrated_cases.json')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, a = load_source(args.source)
    p, entropy, top10 = patch_metrics(a)
    normalized_entropy = entropy / np.log(196)
    route = np.array([r['routed_to_6h'] == '1' for r in rows])
    groups = {
        'exited_at_3h': ~route,
        'routed_to_6h': route,
        'routed_corrected': route & np.array([r['group'] == 'corrected' for r in rows]),
        'routed_regressed': route & np.array([r['group'] == 'regressed' for r in rows]),
    }
    assert [int(m.sum()) for m in groups.values()] == [31996,18004,4330,945]
    metrics = []
    for name, mask in groups.items():
        r = {'group': name, 'samples': int(mask.sum())}
        for label, values in [('entropy_nats', entropy), ('normalized_entropy', normalized_entropy), ('top10_mass', top10)]:
            for stat, fn in [('mean', np.mean), ('median', np.median), ('std', np.std)]:
                r[f'{label}_{stat}'] = float(fn(values[mask]))
        metrics.append(r)
    write_csv(out / 'group_statistics.csv', metrics)
    write_csv(out / 'image_metrics.csv', [dict(r, attention_entropy_nats=float(entropy[i]),
        normalized_attention_entropy=float(normalized_entropy[i]), top10_patch_mass=float(top10[i]),
        source_cls_self_attention=float(a[i,0])) for i,r in enumerate(rows)])
    summary = {'source': str(args.source), 'samples': len(rows), 'groups': metrics,
        'definitions': {'attention': 'Final narrow block, mean over 3 heads; remove CLS column then normalize 196 patch weights to sum to 1.',
         'entropy': '-sum(p*ln(p)), nats; maximum ln(196); normalized entropy H/ln(196). This is NOT classifier output entropy.',
         'top10_mass': 'Sum of the ten largest normalized patch weights.',
         'histogram': 'One figure with shared-bin density histograms; two metrics and two comparisons.'},
        'source_sha256': {n:hashlib.sha256((args.source/n).read_bytes()).hexdigest() for n in ['images.csv','narrow_cls_attention.npy']},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out / 'summary.json', summary)
    print(json.dumps(metrics, indent=2), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    fig, axs = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for j,(values,label,edges) in enumerate([(entropy,'Patch attention entropy (nats)',np.linspace(0,np.log(196),45)),
                                           (top10,'Attention mass in top 10 patches',np.linspace(0,1,45))]):
        for i,pair in enumerate([('exited_at_3h','routed_to_6h'),('routed_corrected','routed_regressed')]):
            for name,color in zip(pair,['#187b8c','#d95f02']):
                mask=groups[name]
                axs[i,j].hist(values[mask],bins=edges,density=True,histtype='step',lw=2,color=color,
                              label=f'{name} (n={mask.sum():,})')
            axs[i,j].set(xlabel=label,ylabel='Density')
            axs[i,j].legend(fontsize=8)
            axs[i,j].grid(alpha=.15)
    fig.suptitle('3H final-block CLS attention | patch-only normalization | ImageNet-1K 50,000 images')
    fig.savefig(out/'histogram.png')
    plt.close(fig)
    cases = json.loads(args.cases.read_text())
    if len(cases)!=8:
        raise ValueError('Expected report section 6 eight cases')
    selected=[]
    # One shared scale across cases: no per-image max normalization.
    vmax=float(max(p[int(c['attention_row'])].max() for c in cases))
    norm=Normalize(0,vmax)
    fig,axes=plt.subplots(4,2,figsize=(10,18),layout='constrained')
    (out/'overlays').mkdir(exist_ok=True)
    for ax,c in zip(axes.flat,cases):
        row=int(c['attention_row']); source=rows[row]
        for k in ['dataset_index','image_path','target','narrow_prediction','wide_prediction']:
            if c[k]!=source[k]: raise ValueError('Case identity mismatch')
        imgpath=ROOT/'baseline'/c['published_input']
        if hashlib.sha256(imgpath.read_bytes()).hexdigest()!=c['input_sha256']:
            raise ValueError('Stored model input checksum mismatch')
        im=np.asarray(Image.open(imgpath).convert('RGB'))
        assert im.shape==(224,224,3)
        grid=p[row].reshape(14,14)
        heat=np.repeat(np.repeat(grid,16,0),16,1)
        rgba=plt.get_cmap('magma')(norm(heat))
        overlay=(.6*im/255+.4*rgba[:,:,:3])
        slug=Path(c['published_input']).stem.removesuffix('_input')
        png=out/'overlays'/f'{slug}_attention.png'
        plt.imsave(png,overlay)
        np.save(out/'overlays'/f'{slug}_attention.npy',grid)
        ax.imshow(overlay)
        ax.set_title(f"{c['target_name'].split(',')[0]} [{c['target']}] -> {c['wide_prediction_name'].split(',')[0]} [{c['wide_prediction']}]\n"
                     f"H={entropy[row]:.3f}; top10={top10[row]:.1%}; ID={c['original_validation_id']}",fontsize=10)
        ax.axis('off')
        selected.append({'dataset_index':int(c['dataset_index']),'attention_row':row,'title':c['target_name']+' -> '+c['wide_prediction_name'],
                         'overlay':str(png.relative_to(out)),'entropy_nats':float(entropy[row]),'top10_mass':float(top10[row])})
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='magma'),ax=axes.ravel().tolist(),shrink=.5,
                 label='Patch attention probability (shared scale; overlay alpha=0.4)')
    fig.suptitle('3H CLS attention on actual 224 x 224 input | 14 x 14 cells, no smoothing')
    fig.savefig(out/'attention_maps.png')
    plt.close(fig)
    write_json(out/'overlay_cases.json',{'shared_vmin':0,'shared_vmax':vmax,'heatmap_alpha':.4,'cases':selected})
    print(f'Results: {out}',flush=True)


if __name__=='__main__':
    main()
