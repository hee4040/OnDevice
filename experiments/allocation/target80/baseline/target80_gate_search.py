#!/usr/bin/env python3
"""Calibration-only joint exit/retention search, with explicit extra-cost objectives."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from analyze_adaptive_drop import validate_cache,save_json,save_csv
from plot_h1_diagnosis import NARROW_MACS,wide_macs

RATIOS=[.25,.5,.75,.9,1.]
COSTS=np.array([wide_macs({'ratio':r,'mode':'drop','k':0}) for r in RATIOS],dtype=np.int64)

def boundary(x):return '+inf' if np.isposinf(x) else float(x)
def number(x):return np.inf if x=='+inf' else float(x)

def codes_for(entropy,policy):
    e=np.asarray(entropy,dtype=np.float64)
    if policy['kind']=='fixed':codes=np.full(len(e),RATIOS.index(policy['ratio']),dtype=np.int64)
    else:codes=np.where(e<number(policy['t1']),0,np.where(e<number(policy['t2']),1,2))
    return np.where(e<number(policy['gate']),-1,codes)

def evaluate(e,narrow,wide,policy):
    codes=codes_for(e,policy);r=codes>=0;correct=narrow.copy();correct[r]=wide[np.flatnonzero(r),codes[r]]
    cost=int(NARROW_MACS)*len(e)+int(COSTS[codes[r]].sum())
    return int(correct.sum()),cost,codes,correct

def candidates(e,narrow,wide,grid=None):
    grid=np.r_[np.arange(48,dtype=float)/20,np.inf] if grid is None else np.array(grid)
    order=np.argsort(e,kind='stable');e=e[order];all_correct=np.column_stack([narrow,wide])[order].astype(np.int64)
    prefix=np.vstack([np.zeros(6,dtype=np.int64),all_correct.cumsum(0)])
    pos=np.searchsorted(e,grid,side='left');n=len(e);result=[]
    for i,gate in enumerate(grid):
        start=int(pos[i])
        for code,ratio in enumerate(RATIOS):
            score=int(prefix[start,0]+prefix[n,code+1]-prefix[start,code+1])
            cost=int(NARROW_MACS)*n+(n-start)*int(COSTS[code])
            result.append({'kind':'fixed','gate':boundary(gate),'ratio':ratio,'correct':score,'cost':cost})
        for j in range(i,len(grid)):
            middle=int(pos[j])
            for k in range(j,len(grid)):
                end=int(pos[k]);score=int(prefix[start,0]+prefix[middle,1]-prefix[start,1]+prefix[end,2]-prefix[middle,2]+prefix[n,3]-prefix[end,3])
                cost=int(NARROW_MACS)*n+int(np.array([middle-start,end-middle,n-end])@COSTS[:3])
                result.append({'kind':'monotone','gate':boundary(gate),'t1':boundary(grid[j]),'t2':boundary(grid[k]),'correct':score,'cost':cost})
    return result

def choose(rows,budget,n):
    policy_key=lambda r:json.dumps(r,sort_keys=True)
    best=min((r for r in rows if r['cost']<=budget),key=lambda r:(-r['correct'],r['cost'],policy_key(r)))
    selected={'same_budget':best}
    for percent in [80.,80.2]:
        eligible=[r for r in rows if r['correct']>=int(np.ceil(n*percent/100))]
        selected[f'target_{percent:.1f}']=min(eligible,key=lambda r:(r['cost'],-r['correct'],policy_key(r))) if eligible else None
    return selected

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];protocol=json.loads((root/'configs/target80_20261108.json').read_text());save_json(out/'protocol.json',protocol)
    path=args.source/'cache/per_image.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['cache_sha256']:raise ValueError('Cache mismatch')
    cache=torch.load(path,weights_only=True);validate_cache(cache)
    split=json.loads((args.source/'split_ids.json').read_text());order=cache['image_id'].argsort()
    e=cache['entropy'][order].double().numpy();narrow=cache['narrow_correct'][order].numpy()
    wide=np.stack([cache['wide_correct' if r==1 else f'drop_{round(100*r)}_correct'][order].numpy() for r in RATIOS],axis=1)
    cal=np.array(sorted(split['calibration']));baseline={'kind':'fixed','gate':1.,'ratio':.5}
    budget=evaluate(e[cal],narrow[cal],wide[cal],baseline)[1]
    rows=candidates(e[cal],narrow[cal],wide[cal]);selected=choose(rows,budget,len(cal))
    selected={'baseline':baseline,**selected}
    save_json(out/'policies.json',selected) # Lock BEFORE evaluating the old holdout.
    save_json(out/'calibration_candidates.json',rows)
    reports=[]
    for split_name,index in [('calibration',cal),('previously_seen_holdout',np.array(sorted(split['holdout']))),('full_validation',np.arange(50000))]:
        _,limit,_,base_correct=evaluate(e[index],narrow[index],wide[index],baseline)
        for name,p in selected.items():
            if p is None:continue
            correct,cost,codes,corr=evaluate(e[index],narrow[index],wide[index],p)
            reports.append({'split':split_name,'method':name,'n':len(index),'correct':correct,'top1':100*correct/len(index),
                'gmacs':cost/len(index)/1e9,'total_macs':cost,'baseline_macs':limit,'budget_exceeded':cost>limit,
                'cost_change_percent':100*(cost/limit-1),'target80_reached':correct>=.8*len(index),
                'improved':int((corr&~base_correct).sum()),'regressed':int((~corr&base_correct).sum()),
                'stage2_fraction':float((codes>=0).mean())})
    save_csv(out/'comparison.csv',reports);save_json(out/'summary.json',{'policies':selected,'comparison':reports,'candidates':len(rows),
        'scope':protocol['scope'],'source':str(args.source.resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'policies':selected,'comparison':reports},indent=2))

if __name__=='__main__':main()
