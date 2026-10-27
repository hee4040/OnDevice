#!/usr/bin/env python3
"""Individually zero the top-attention input patches and measure fixed top1-top2 logits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
from analyze_attention import ROOT, load_source, patch_metrics, write_json, write_csv


def effect_label(delta, epsilon=1e-4):
    if delta < -epsilon:
        return 'supports_top1_margin'
    if delta > epsilon:
        return 'opposes_top1_margin'
    return 'near_zero'


def masked_batch(image, patch_indices):
    """Two identical controls plus independent single-patch interventions."""
    if tuple(image.shape) != (3,224,224):
        raise ValueError('Expected normalized [3,224,224] image')
    if any(not 0 <= int(k) < 196 for k in patch_indices):
        raise ValueError('Invalid patch index')
    batch=image.unsqueeze(0).repeat(len(patch_indices)+2,1,1,1)
    for i,k in enumerate(patch_indices,2):
        y,x=divmod(int(k),14)
        batch[i,:,y*16:(y+1)*16,x*16:(x+1)*16]=0
    return batch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--samples',type=int,default=200)
    ap.add_argument('--seed',type=int,default=20261027)
    args=ap.parse_args()
    rows,a=load_source(args.source)
    p,_,_=patch_metrics(a)
    population=sorted([r for r in rows if r['routed_to_6h']=='1'],key=lambda r:int(r['dataset_index']))
    if not 1<=args.samples<=len(population):
        ap.error('Invalid sample size')
    selected=random.Random(args.seed).sample(population,args.samples)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'sample_indices.json',{'population':len(population),'seed':args.seed,
        'method':'uniform random sample without replacement; routed population sorted by dataset_index',
        'dataset_indices':[int(r['dataset_index']) for r in selected],
        'original_groups':dict(Counter(r['group'] for r in selected))})
    env=json.loads((args.source/'environment.json').read_text())
    repo=Path(env['arguments']['repo'])
    checkpoint=Path(env['arguments']['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=env['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    sys.path.insert(0,str(repo))
    import torch
    import timm
    from timm.data import create_transform
    from PIL import Image
    from validate import _load_eval_checkpoint, RLDropping
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if not Path(timm.__file__).resolve().is_relative_to(repo): raise RuntimeError('Wrong timm')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    model=timm.create_model('thinkingvit',pretrained=False,thinking_stages=[3,6])
    _load_eval_checkpoint(model,str(checkpoint),use_ema=True)
    RLDropping(model,used_heads=6,max_heads=6,ema=False)
    model=model.cuda().eval()
    model.p=model._stage_ratio(3)
    transform=create_transform(**env['preprocessing'],is_training=False)
    data=Path(env['arguments']['data_dir'])
    records=[]
    image_records=[]
    noop=[]
    with torch.no_grad():
        for r in selected:
            weights=p[int(r['attention_row'])]
            patches=np.argsort(-weights,kind='stable')[:10]
            with Image.open(data/r['image_path']) as im:
                x=transform(im.convert('RGB'))
            batch=masked_batch(x,patches).cuda()
            logits=model.forward_head(model.forward_features(batch,model.p)).float().cpu()
            if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite logits')
            original=logits[0]
            top=original.topk(2).indices.tolist()
            c1,c2=top
            base=float(original[c1]-original[c2])
            noop.append(float((logits[1,c1]-logits[1,c2])-(original[c1]-original[c2])))
            changed=int(c1!=int(r['narrow_prediction']))
            image_records.append({**r,'fixed_class_1':c1,'fixed_class_2':c2,'unmasked_margin':base,
                                  'source_top1_changed':changed,'no_op_margin_delta':noop[-1]})
            for j,k in enumerate(patches,2):
                delta1=float(logits[j,c1]-original[c1]); delta2=float(logits[j,c2]-original[c2])
                masked=float(logits[j,c1]-logits[j,c2])
                records.append({'dataset_index':int(r['dataset_index']),'attention_row':int(r['attention_row']),
                    'image_path':r['image_path'],'group':r['group'],'target':int(r['target']),
                    'fixed_class_1':c1,'fixed_class_2':c2,'source_top1_changed':changed,
                    'attention_rank':j-1,'patch_index':int(k),'patch_y':int(k)//14,'patch_x':int(k)%14,
                    'attention_probability':float(weights[k]),'unmasked_logit_1':float(original[c1]),
                    'unmasked_logit_2':float(original[c2]),'masked_logit_1':float(logits[j,c1]),'masked_logit_2':float(logits[j,c2]),
                    'delta_logit_1':delta1,'delta_logit_2':delta2,'unmasked_margin':base,'masked_margin':masked,
                    'delta_margin':masked-base,'masked_top1':int(logits[j].argmax())})
            if len(image_records)%20==0:
                write_csv(out/'patch_effects.partial.csv',records)
                print(f'Occluded {len(image_records)}/{len(selected)} images ({len(records)} independent masks)',flush=True)
    epsilon=max(1e-4,10*max(map(abs,noop)))
    for r in records:
        r['effect']=effect_label(r['delta_margin'],epsilon)
        r['strict_sign_effect']=effect_label(r['delta_margin'],0)
    for r in image_records:
        parts=[d for d in records if d['dataset_index']==int(r['dataset_index'])]
        for name in ['supports_top1_margin','opposes_top1_margin','near_zero']:
            r[name+'_fraction']=sum(d['effect']==name for d in parts)/10
        r['mean_delta_margin']=float(np.mean([d['delta_margin'] for d in parts]))
    def summarize(subset):
        ids={r['dataset_index'] for r in subset}
        return {'images':len(ids),'patches':len(subset),'counts':dict(Counter(r['effect'] for r in subset)),
            'strict_sign_counts':dict(Counter(r['strict_sign_effect'] for r in subset)),
            'mean_delta_margin':float(np.mean([r['delta_margin'] for r in subset])) if subset else None,
            'median_delta_margin':float(np.median([r['delta_margin'] for r in subset])) if subset else None,
            'mean_abs_delta_margin':float(np.mean([abs(r['delta_margin']) for r in subset])) if subset else None,
            'masked_top1_flips':sum(r['masked_top1']!=r['fixed_class_1'] for r in subset),
            'opposing_patches_with_both_logits_decreasing':sum(r['effect']=='opposes_top1_margin' and r['delta_logit_1']<0 and r['delta_logit_2']<0 for r in subset)}
    summary={'source':str(args.source),'samples':len(selected),'seed':args.seed,'epsilon':epsilon,
        'max_abs_noop_margin_delta':max(map(abs,noop)),
        'source_top1_changes':sum(r['source_top1_changed'] for r in image_records),
        'all':summarize(records),'groups':{g:summarize([r for r in records if r['group']==g]) for g in sorted({r['group'] for r in records})},
        'fraction_distribution':{k:{'mean':float(np.mean([r[k+'_fraction'] for r in image_records])),
                                   'median':float(np.median([r[k+'_fraction'] for r in image_records]))}
                                 for k in ['supports_top1_margin','opposes_top1_margin','near_zero']},
        'protocol':{'patches':'Top 10 from saved pre-intervention attention; 16x16 pixels, 14x14 row-major grid.',
            'intervention':'Each patch independently set to zero AFTER official normalization. Other 195 patches unchanged. This corresponds to RGB normalization mean, not black.',
            'forward':'Only 3H; fixed original unmasked top1 and top2 classes for all ten masks; original routing population retained.',
            'batch':'12 inputs per image: 2 identical unmasked controls and 10 independent masks.',
            'interpretation':'Negative delta margin supports original top1 relative to top2; positive opposes that margin. Not proof of ground-truth evidence or psychological confusion. Last-block attention is used to select input-level interventions; not token-update intervention.',
            'numerics':'float32; AMP off; TF32 allowed; epsilon=max(1e-4,10*max_abs_noop_delta); strict signs also reported.'},
        'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
                       'preprocessing':env['preprocessing'],'checkpoint_sha256':env['checkpoint_sha256'],'repo_commit':env['repo_commit']},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256((args.source/n).read_bytes()).hexdigest() for n in ['images.csv','narrow_cls_attention.npy']}}
    write_csv(out/'patch_effects.csv',records)
    write_csv(out/'image_effects.csv',image_records)
    write_json(out/'summary.json',summary)
    (out/'patch_effects.partial.csv').unlink(missing_ok=True)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Results: {out}',flush=True)


if __name__=='__main__':
    main()
