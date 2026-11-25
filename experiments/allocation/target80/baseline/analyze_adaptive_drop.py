#!/usr/bin/env python3
"""Preregistered, exact-budget calibration/holdout analysis of a fixed student's cache."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_drop import ratio_codes, RATIOS
from plot_h1_diagnosis import NARROW_MACS, wide_macs

ROOT = Path(__file__).resolve().parents[1]
WIDE_COSTS = np.array([wide_macs({'ratio':r,'mode':'drop','k':0}) for r in RATIOS],dtype=np.int64)
KEYS = ['narrow_correct','wide_correct','drop_25_correct','drop_50_correct','drop_75_correct']


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def validate_cache(cache, expected_n=50000):
    e=cache['entropy']
    if not isinstance(e,torch.Tensor) or e.shape!=(expected_n,) or not e.dtype.is_floating_point or not torch.isfinite(e).all():
        raise ValueError('Invalid entropy cache')
    for key in KEYS:
        if cache[key].shape!=(expected_n,) or cache[key].dtype!=torch.bool:
            raise ValueError(f'Invalid bool array: {key}')
    ids=cache['image_id']
    if ids.dtype!=torch.int64 or ids.shape!=(expected_n,) or not torch.equal(ids.sort().values,torch.arange(expected_n)):
        raise ValueError('Cache needs unique complete ImageFolder IDs, not inferred legacy ordering')
    paths=cache['image_path']
    if len(paths)!=expected_n or len(set(paths))!=expected_n or any(not isinstance(p,str) or Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
        raise ValueError('Invalid relative image paths')
    t=cache['target']
    if t.dtype!=torch.int64 or t.shape!=(expected_n,) or (t<0).any() or (t>=1000).any():
        raise ValueError('Invalid targets')


def arrays(cache, index):
    return {'ids':cache['image_id'][index].numpy(), 'entropy':cache['entropy'][index].double().numpy(),
            'narrow':cache['narrow_correct'][index].numpy(),
            'correct':np.stack([cache[f'drop_{round(r*100)}_correct'][index].numpy() for r in RATIOS],axis=1)}


def outcomes(data, codes):
    if codes.shape != data['entropy'].shape or not np.isin(codes, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid ratio codes')
    routed=data['entropy']>=1.0
    if not np.array_equal(codes>=0,routed):
        raise ValueError('Policy changed the fixed routing gate')
    correct=data['narrow'].copy()
    correct[routed]=data['correct'][np.flatnonzero(routed),codes[routed]]
    total_macs=int(NARROW_MACS)*len(codes)+int(WIDE_COSTS[codes[routed]].sum())
    return correct,total_macs


def fixed_codes(data):
    return np.where(data['entropy']>=1,1,-1)


def oracle(data):
    """Exact optimum for unit rewards with three fixed, increasing costs."""
    routed=data['entropy']>=1
    codes=np.where(routed,0,-1)
    budget=outcomes(data,fixed_codes(data))[1]
    _,spent=outcomes(data,codes)
    upgrades=[]
    for i in np.flatnonzero(routed & ~data['correct'][:,0]):
        choices=np.flatnonzero(data['correct'][i])
        if len(choices):
            code=int(choices[0])
            upgrades.append((int(WIDE_COSTS[code]-WIDE_COSTS[0]),int(data['ids'][i]),i,code))
    for cost,_,i,code in sorted(upgrades):
        if spent+cost>budget:
            break
        spent+=cost;codes[i]=code
    result,check=outcomes(data,codes)
    if check!=spent or spent>budget or result.sum()<outcomes(data,fixed_codes(data))[0].sum():
        raise AssertionError('Invalid oracle bound')
    return codes


def boundary_json(v):
    return '-inf' if v==-np.inf else '+inf' if v==np.inf else float(v)


def select_policy(data):
    """Only calibration labels enter this function; integer MACs enforce exact budgets."""
    routed=data['entropy']>=1
    order=np.argsort(data['entropy'][routed],kind='stable')
    entropy=data['entropy'][routed][order]
    correct=data['correct'][routed][order].astype(np.int64)
    prefix=np.vstack([np.zeros((1,3),dtype=np.int64),correct.cumsum(0)])
    quantiles=torch.quantile(torch.from_numpy(entropy),torch.linspace(0,1,101,dtype=torch.float64)).numpy() if len(entropy) else []
    boundaries=np.unique(np.r_[-np.inf,quantiles,np.inf])
    positions=np.searchsorted(entropy,boundaries,side='left')
    easy_correct=int(data['narrow'][~routed].sum())
    budget=outcomes(data,fixed_codes(data))[1]
    best=None;candidates=0;feasible=0
    for ai,t1 in enumerate(boundaries):
        i=int(positions[ai])
        for aj in range(ai,len(boundaries)):
            t2=boundaries[aj];j=int(positions[aj]);candidates+=1
            counts=np.array([i,j-i,len(entropy)-j],dtype=np.int64)
            macs=int(NARROW_MACS)*len(routed)+int(counts@WIDE_COSTS)
            if macs>budget:continue
            feasible+=1
            score=easy_correct+int(prefix[i,0]+prefix[j,1]-prefix[i,1]+prefix[-1,2]-prefix[j,2])
            key=(-score,macs,float(t1),float(t2))
            if best is None or key<best[0]:best=(key,counts.tolist())
    if best is None:raise AssertionError('Fixed50 must always be feasible')
    key,counts=best
    return {'kind':'entropy','threshold':1.0,'t1':boundary_json(key[2]),'t2':boundary_json(key[3]),
            'calibration_correct':-key[0],'calibration_samples':len(routed),
            'calibration_total_macs':key[1],'calibration_budget_macs':budget,'routed_ratio_counts':counts,
            'candidate_pairs':candidates,'feasible_pairs':feasible,
            'tie_break':'correct count desc, integer MACs asc, t1 asc, t2 asc',
            'fixed50_included':'t1=-inf,t2=+inf'}


def apply_policy(data,policy):
    return ratio_codes(torch.from_numpy(data['entropy']),policy).numpy()


def compare_row(data,codes,split,method):
    correct,macs=outcomes(data,codes)
    baseline,budget=outcomes(data,fixed_codes(data));routed=codes>=0
    return {'model':'fixed_finetuned_student','method':method,'split':split,'N':len(codes),
            'routed_N':int(routed.sum()),'top1':100*float(correct.mean()),
            'routed_top1':100*float(correct[routed].mean()) if routed.any() else None,
            'delta_fixed50_pp':100*float(correct.mean()-baseline.mean()),'ci_low_pp':None,'ci_high_pp':None,
            'gmacs_per_image':macs/len(codes)/1e9,'budget_gmacs_per_image':budget/len(codes)/1e9,
            'total_macs':macs,'budget_total_macs':budget,'budget_exceeded':bool(macs>budget),
            'exit_count':int((codes==-1).sum()),'ratio25_count':int((codes==0).sum()),
            'ratio50_count':int((codes==1).sum()),'ratio75_count':int((codes==2).sum())}


def paired_interval(adaptive,fixed,seed=0,repeats=2000):
    delta=adaptive.astype(np.int8)-fixed.astype(np.int8)
    rng=np.random.default_rng(seed)
    means=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(repeats)])*100
    return np.quantile(means,[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--student',type=Path,required=True,help='New identified cache directory from evaluate_drop.py')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--split-seed',type=int,default=20261107)
    ap.add_argument('--bootstrap-seed',type=int,default=0)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'policy.json').exists():raise ValueError('Policy already locked; use a new output for a deliberate rerun')
    protocol=json.loads((ROOT/'configs/adaptive_drop_20261107.json').read_text())
    save_json(out/'analysis_config.json',dict(protocol,split_seed=args.split_seed,bootstrap_seed=args.bootstrap_seed))
    meta=json.loads((args.student/'cache_metadata.json').read_text())
    if meta.get('weights_sha256')!=protocol['expected_student_sha256']:
        raise ValueError('Wrong student checkpoint; do not replace with official weights or retrain')
    if meta.get('official_checkpoint_sha256')!=protocol['expected_official_sha256'] or meta.get('identity_schema')!=1:
        raise ValueError('Wrong official checkpoint or legacy unidentified cache')
    if meta.get('batch_size') != protocol['batch_size'] or meta.get('evaluation_seed') != protocol['evaluation_seed']:
        raise ValueError('Cache batch size/seed differs from the preregistered evaluation')
    if not meta.get('full_path_matches_official') or meta.get('samples') != protocol['samples']:
        raise ValueError('Incomplete or unverified cache')
    transform = meta.get('transform_config', {})
    required_transform = {'input_size': [3, 224, 224], 'interpolation': 'bicubic', 'crop_pct': 0.9,
                          'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    if any(transform.get(key) != value for key, value in required_transform.items()):
        raise ValueError('Cache preprocessing differs from the official baseline')
    if hashlib.sha256((args.student/'per_image.pt').read_bytes()).hexdigest()!=meta['cache_sha256']:
        raise ValueError('Cache checksum mismatch')
    cache=torch.load(args.student/'per_image.pt',map_location='cpu',weights_only=True)
    validate_cache(cache)
    ordered=cache['image_id'].argsort()
    perm=torch.randperm(50000,generator=torch.Generator().manual_seed(args.split_seed))
    splits={'calibration':ordered[perm[:25000]],'holdout':ordered[perm[25000:]],'all':ordered}
    save_json(out/'split_ids.json',{'split_seed':args.split_seed,
        **{k:cache['image_id'][v].tolist() for k,v in splits.items() if k!='all'}})
    calibration=arrays(cache,splits['calibration'])
    policy=select_policy(calibration)
    policy['student_weights_sha256']=meta['weights_sha256']
    policy['cache_sha256']=meta['cache_sha256']
    # Lock before touching holdout outcomes or oracle allocations.
    save_json(out/'policy.json',policy)
    comparisons=[];transitions=[];oracle_records={}
    for split,index in splits.items():
        data=arrays(cache,index)
        fixed=fixed_codes(data);adaptive=apply_policy(data,policy)
        variants={'fixed50':fixed,'adaptive':adaptive}
        if split in ('all','holdout'):
            variants['oracle']=oracle(data)
            oracle_records[split]={'diagnostic_only':True,'image_ids':data['ids'].tolist(),
                                   'ratio_codes':variants['oracle'].tolist(),**compare_row(data,variants['oracle'],split,'oracle')}
        for method,codes in variants.items():
            row=compare_row(data,codes,split,method)
            correct,_=outcomes(data,codes)
            if split=='holdout' and method=='adaptive':
                ci=paired_interval(correct,outcomes(data,fixed)[0],args.bootstrap_seed)
                row['ci_low_pp'],row['ci_high_pp']=ci
                success=not row['budget_exceeded'] and ci[0]>0
                verdict='improved_within_budget' if success else 'budget_exceeded' if row['budget_exceeded'] else 'improvement_not_confirmed'
            comparisons.append(row)
            if method!='oracle':
                for scope,mask in [('all',np.ones(len(codes),dtype=bool)),('routed',codes>=0)]:
                    n=data['narrow'][mask];w=correct[mask]
                    transitions.append({'split':split,'method':method,'scope':scope,'N':int(mask.sum()),
                        'both_correct':int((n&w).sum()),'corrected':int((~n&w).sum()),
                        'regressed':int((n&~w).sum()),'both_wrong':int((~n&~w).sum())})
    save_json(out/'oracle.json',oracle_records)
    save_csv(out/'comparison.csv',comparisons);save_csv(out/'transitions.csv',transitions)
    save_json(out/'analysis_summary.json',{'verdict':verdict,'policy':policy,'comparisons':comparisons,
        'fixed50_full_reproduced_top1':next(r['top1'] for r in comparisons if r['split']=='all' and r['method']=='fixed50'),
        'published_fixed50_reference':79.65,'cost_definition':'Upstream analytic MACs; excludes normalization, softmax and selector overhead; not latency.',
        'mobile_latency':'unmeasured','cache_metadata':meta})
    print(json.dumps({'verdict':verdict,'policy':policy,'holdout':[r for r in comparisons if r['split']=='holdout']},indent=2))


if __name__=='__main__':main()
