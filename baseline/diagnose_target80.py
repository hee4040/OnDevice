#!/usr/bin/env python3
"""Post-hoc crossfit robustness and attribution; no primary policy retuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from target80_gate_search import candidates,evaluate,choose,RATIOS
from analyze_adaptive_drop import save_json,save_csv
from evaluate_nonmonotone_entropy import stratified_paired_ci

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True)
    args=ap.parse_args();out=args.experiment/'diagnostics';out.mkdir(exist_ok=False)
    save_json(out/'protocol.json',{'scope':'Post-hoc repeated crossfit on all 50000 already explored images; not independent confirmation; cache based',
        'seeds':[20261108,20261109],'folds':5,'budget_fractions':[1.,.98],
        'selection':'Frozen candidate grid; maximize training correct count within training cost; no retuning on evaluation fold',
        'ci':'Conditional paired bootstrap with fixed OOF predictions; excludes selection/repeated-exploration uncertainty'})
    cache=torch.load(args.source/'cache/per_image.pt',weights_only=True);order=cache['image_id'].argsort()
    e=cache['entropy'][order].double().numpy();narrow=cache['narrow_correct'][order].numpy()
    wide=np.stack([cache['wide_correct' if r==1 else f'drop_{round(100*r)}_correct'][order].numpy() for r in RATIOS],axis=1)
    baseline={'kind':'fixed','gate':1.,'ratio':.5};_,budget,_,base_corr=evaluate(e,narrow,wide,baseline)
    fold_rows=[];pooled=[];policies=[]
    for seed in [20261108,20261109]:
        folds=np.zeros(50000,dtype=np.int64)
        for f,idx in enumerate(np.array_split(np.random.default_rng(seed).permutation(50000),5)):folds[idx]=f
        predictions={b:np.zeros(50000,dtype=bool) for b in [1.,.98]};costs={b:0 for b in predictions}
        for f in range(5):
            train=folds!=f;test=~train
            _,train_budget,_,_=evaluate(e[train],narrow[train],wide[train],baseline)
            _,test_budget,_,_=evaluate(e[test],narrow[test],wide[test],baseline)
            options=candidates(e[train],narrow[train],wide[train])
            for b in [1.,.98]:
                policy=choose(options,train_budget*round(100*b)//100,int(train.sum()))['same_budget']
                policies.append({'seed':seed,'fold':f+1,'budget_fraction':b,'policy':policy})
                save_json(out/'policies.partial.json',policies)
                correct,cost,_,corr=evaluate(e[test],narrow[test],wide[test],policy)
                predictions[b][test]=corr;costs[b]+=cost
                fold_rows.append({'seed':seed,'fold':f+1,'budget_fraction':b,'n':int(test.sum()),'top1':100*correct/test.sum(),
                    'gmacs':cost/test.sum()/1e9,'cost_change_percent':100*(cost/test_budget-1),'budget_exceeded':cost>test_budget})
            print(f'Post-hoc crossfit {seed} fold {f+1}/5',flush=True)
        for b,corr in predictions.items():
            lo,hi=stratified_paired_ci(corr,base_corr,folds,seed=0,repeats=5000)
            pooled.append({'seed':seed,'budget_fraction':b,'top1':100*float(corr.mean()),'gmacs':costs[b]/50000/1e9,
                'delta_baseline_pp':100*float((corr.astype(np.int8)-base_corr.astype(np.int8)).mean()),'ci_low_pp':lo,'ci_high_pp':hi,
                'cost_change_percent':100*(costs[b]/budget-1),'budget_exceeded':costs[b]>budget,
                'ceiling_exceeded_folds':sum(r['budget_exceeded'] for r in fold_rows if r['seed']==seed and r['budget_fraction']==b)})
        torch.save({'folds':torch.from_numpy(folds),'correct':{str(b):torch.from_numpy(c) for b,c in predictions.items()}},out/f'oof_{seed}.pt')
    save_csv(out/'crossfit_folds.csv',fold_rows);save_csv(out/'crossfit_pooled.csv',pooled);save_json(out/'crossfit_policies.json',policies)
    actual=torch.load(args.experiment/'gpu_accuracy/per_image.pt',weights_only=True);idx=actual['image_id'].argsort()
    a=actual['methods']['same_budget']['correct'][idx].numpy();base=actual['methods']['baseline']['correct'][idx].numpy()
    regions=[('entropy < 0.8',e<.8),('0.8 <= entropy < 1.0',(e>=.8)&(e<1.)),
             ('1.0 <= entropy < 1.25',(e>=1.)&(e<1.25)),('entropy >= 1.25',e>=1.25)]
    attribution=[]
    for label,mask in regions:
        attribution.append({'region':label,'n':int(mask.sum()),'baseline_correct':int(base[mask].sum()),'new_correct':int(a[mask].sum()),
            'improved':int((a[mask]&~base[mask]).sum()),'regressed':int((~a[mask]&base[mask]).sum()),'net_correct':int(a[mask].sum())-int(base[mask].sum())})
    save_csv(out/'actual_region_attribution.csv',attribution)
    split=json.loads((args.source/'split_ids.json').read_text());cal=np.isin(np.arange(50000),split['calibration']);paired=[]
    for label,mask in [('calibration',cal),('previously_seen_holdout',~cal),('full_validation',np.ones(50000,dtype=bool))]:
        lo,hi=stratified_paired_ci(a[mask],base[mask],np.zeros(int(mask.sum()),dtype=int),seed=0,repeats=5000)
        paired.append({'split':label,'delta_pp':100*float((a[mask].astype(np.int8)-base[mask].astype(np.int8)).mean()),'ci_low_pp':lo,'ci_high_pp':hi})
    save_csv(out/'actual_paired_intervals.csv',paired)
    save_json(out/'summary.json',{'crossfit':pooled,'attribution':attribution,'paired_intervals':paired})
    print(json.dumps({'crossfit':pooled,'attribution':attribution,'paired_intervals':paired},indent=2))

if __name__=='__main__':main()
