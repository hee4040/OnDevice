#!/usr/bin/env python3
"""Replay all frozen policies on all 50k images with real conditional 6H execution."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from evaluate_drop import add_paths,load_model,ROOT
from selective_refinement import forward_narrow_with_attention
from target80_runtime import forward_target80
from target80_gate_search import RATIOS,COSTS,codes_for
from analyze_adaptive_drop import NARROW_MACS,save_json,save_csv

def main():
    ap=argparse.ArgumentParser(description=__doc__);add_paths(ap)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--experiment',type=Path,required=True);args=ap.parse_args();out=args.experiment/'gpu_accuracy';out.mkdir(exist_ok=False)
    config=json.loads((ROOT/'configs/target80_20261108.json').read_text())
    if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=config['student_sha256']:raise ValueError('Wrong student')
    cache=torch.load(args.source/'cache/per_image.pt',weights_only=True);split=json.loads((args.source/'split_ids.json').read_text())
    policies=json.loads((args.experiment/'gate_search/policies.json').read_text());policies={k:v for k,v in policies.items() if v is not None}
    save_json(out/'locked_policies.json',policies)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    model=load_model(args.repo.resolve(),args.checkpoint.resolve(),args.weights)
    from timm.data import create_transform,resolve_data_config
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader,Subset
    transform=resolve_data_config({},model=model,use_test_size=True)
    original=json.loads((args.source/'cache/cache_metadata.json').read_text())
    if json.loads(json.dumps(transform))!=original['transform_config']:raise ValueError('Preprocessing changed')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=original['official_checkpoint_sha256']:raise ValueError('Wrong official checkpoint')
    if hashlib.sha256((args.source/'cache/per_image.pt').read_bytes()).hexdigest()!=config['cache_sha256']:raise ValueError('Wrong reference cache')
    dataset=ImageFolder(args.data_dir/'validation',transform=create_transform(**transform,is_training=False))
    ids=cache['image_id'].tolist();root=(args.data_dir/'validation').resolve()
    for row,i in enumerate(ids):
        path,target=dataset.samples[i]
        if str(Path(path).resolve().relative_to(root))!=cache['image_path'][row] or target!=cache['target'][row].item():raise ValueError('Image identity mismatch')
    loader=DataLoader(Subset(dataset,ids),batch_size=64,num_workers=8,pin_memory=True)
    saved={k:{'correct':[],'codes':[],'predicted':[]} for k in policies};offset=0;max_entropy_error=0.;narrow_mismatches=0
    with torch.no_grad():
        for bi,(images,target) in enumerate(loader,1):
            images=images.cuda(non_blocking=True);narrow=forward_narrow_with_attention(model,images)
            narrow_mismatches+=int(((narrow[1].argmax(1).cpu()==target)!=cache['narrow_correct'][offset:offset+len(images)]).sum())
            for name,policy in policies.items():
                logits,codes,e=forward_target80(model,images,policy,narrow)
                predicted=logits.argmax(1).cpu();e=e.cpu()
                max_entropy_error=max(max_entropy_error,float((e-cache['entropy'][offset:offset+len(images)]).abs().max()))
                saved[name]['correct'].append(predicted==target);saved[name]['codes'].append(codes.cpu());saved[name]['predicted'].append(predicted)
            offset+=len(images)
            if bi%100==0:print(f'Actual conditional inference {offset}/50000',flush=True)
    if offset!=50000:raise ValueError('Incomplete evaluation')
    saved={name:{k:torch.cat(v) for k,v in values.items()} for name,values in saved.items()}
    torch.save({'image_id':cache['image_id'],'target':cache['target'],'methods':saved},out/'per_image.pt')
    cal=np.isin(np.array(ids),split['calibration']);reports=[];mismatches={}
    for name,values in saved.items():
        codes=values['codes'].numpy();corr=values['correct'].numpy();expected_codes=codes_for(cache['entropy'].double().numpy(),policies[name])
        expected=cache['narrow_correct'].numpy().copy()
        for code,r in enumerate(RATIOS):
            mask=expected_codes==code;key='wide_correct' if r==1 else f'drop_{round(100*r)}_correct';expected[mask]=cache[key].numpy()[mask]
        mismatches[name]={'routing':int((codes!=expected_codes).sum()),'correctness':int((corr!=expected).sum()),
                          'image_ids':cache['image_id'][torch.from_numpy(corr!=expected)].tolist()}
        for label,mask in [('calibration',cal),('previously_seen_holdout',~cal),('full_validation',np.ones(50000,dtype=bool))]:
            c=codes[mask];n=int(mask.sum());cost=int(NARROW_MACS)*n+int(COSTS[c[c>=0]].sum())
            base_codes=saved['baseline']['codes'].numpy()[mask];base_cost=int(NARROW_MACS)*n+int(COSTS[base_codes[base_codes>=0]].sum())
            base_corr=saved['baseline']['correct'].numpy()[mask];actual=corr[mask]
            reports.append({'split':label,'method':name,'n':n,'correct':int(actual.sum()),'top1':100*float(actual.mean()),
                'gmacs':cost/n/1e9,'total_macs':cost,'baseline_macs':base_cost,'cost_change_percent':100*(cost/base_cost-1),
                'budget_exceeded':cost>base_cost,'stage2_fraction':float((c>=0).mean()),
                'improved':int((actual&~base_corr).sum()),'regressed':int((~actual&base_corr).sum())})
    save_csv(out/'comparison.csv',reports)
    save_json(out/'summary.json',{'status':'complete','comparison':reports,'cache_mismatches':mismatches,
        'max_entropy_error':max_entropy_error,'narrow_correctness_mismatches':narrow_mismatches,'weights_sha256':config['student_sha256'],
        'transform':transform,'gpu':torch.cuda.get_device_name(),'batch_size':64,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_sha256':hashlib.sha256((ROOT/'baseline/target80_runtime.py').read_bytes()).hexdigest(),
        'note':'Actual conditional grouping can differ numerically from all-image counterfactual cache. All mismatches retained; no policy retuning.'})
    print(json.dumps({'comparison':reports,'cache_mismatches':mismatches},indent=2))

if __name__=='__main__':main()
