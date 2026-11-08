#!/usr/bin/env python3
"""Paired batch-one latency for frozen gate policies, including actual dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from evaluate_drop import ROOT,add_paths,load_model
from target80_runtime import forward_target80
from target80_gate_search import COSTS
from analyze_adaptive_drop import NARROW_MACS,save_json

def main():
    ap=argparse.ArgumentParser(description=__doc__);add_paths(ap)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--experiment',type=Path,required=True);args=ap.parse_args();out=args.experiment
    if (out/'latency.json').exists():raise ValueError('Latency already complete')
    protocol=json.loads((ROOT/'configs/target80_20261108.json').read_text())
    if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=protocol['student_sha256']:raise ValueError('Wrong weights')
    policies=json.loads((out/'gate_search/policies.json').read_text());policies={k:v for k,v in policies.items() if v is not None}
    split=json.loads((args.source/'split_ids.json').read_text());perm=torch.randperm(25000,generator=torch.Generator().manual_seed(42))[:1000]
    ids=[split['holdout'][i] for i in perm.tolist()];names=list(policies)
    save_json(out/'latency_protocol.json',{'image_ids':ids,'seed':42,'samples':1000,'warmup':50,'repeats':3,'batch_size':1,
        'orders':[names[r:]+names[:r] for r in range(3)],'policy_selection':'already frozen, same sample for every policy'})
    cache=torch.load(out/'gpu_accuracy/per_image.pt',weights_only=True);rows=cache['image_id'].argsort()[ids]
    source_cache=torch.load(args.source/'cache/per_image.pt',weights_only=True);source_rows=source_cache['image_id'].argsort()[ids]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    model=load_model(args.repo.resolve(),args.checkpoint.resolve(),args.weights)
    from timm.data import create_transform,resolve_data_config
    from torchvision.datasets import ImageFolder
    dataset=ImageFolder(args.data_dir/'validation',transform=create_transform(**resolve_data_config({},model=model,use_test_size=True),is_training=False))
    for i,row in zip(ids,source_rows.tolist()):
        path,target=dataset.samples[i]
        if str(Path(path).resolve().relative_to((args.data_dir/'validation').resolve()))!=source_cache['image_path'][row] or target!=source_cache['target'][row].item():raise ValueError('Dataset identity mismatch')
    inputs=[dataset[i][0][None] for i in ids];records=[]
    with torch.no_grad():
        for repeat in range(3):
            for name in names[repeat:]+names[:repeat]:
                for image in inputs[:50]:
                    x=image.cuda();result=forward_target80(model,x,policies[name]);del x,result
                torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[];pred=[];codes=[]
                for image in inputs:
                    x=image.cuda();torch.cuda.synchronize();start=time.perf_counter()
                    logits,c,e=forward_target80(model,x,policies[name]);torch.cuda.synchronize()
                    times.append(1000*(time.perf_counter()-start));pred.append(logits.argmax(1).item());codes.append(c.item());del x,logits,c,e
                reference=cache['methods'][name];actual_correct=np.array(pred)==cache['target'][rows].numpy()
                row={'method':name,'repeat':repeat+1,'image_ids':ids,'times_ms':times,'predicted':pred,'codes':codes,
                    'mean_ms':float(np.mean(times)),'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'code_mismatches_vs_batch64':int((np.array(codes)!=reference['codes'][rows].numpy()).sum()),
                    'correctness_mismatches_vs_batch64':int((actual_correct!=reference['correct'][rows].numpy()).sum())}
                records.append(row);save_json(out/'latency.partial.json',records);print(f'{name} repeat {repeat+1}: {row["mean_ms"]:.3f} ms',flush=True)
    summary={}
    for name in names:
        rs=[r for r in records if r['method']==name];times=np.array([r['times_ms'] for r in rs]);codes=np.array([r['codes'] for r in rs]).ravel()
        summary[name]={'mean_ms':float(times.mean()),'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
            'repeat_means':[r['mean_ms'] for r in rs],'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rs),
            'code_mismatches':[r['code_mismatches_vs_batch64'] for r in rs],
            'correctness_mismatches':[r['correctness_mismatches_vs_batch64'] for r in rs],
            'sample_gmacs':(int(NARROW_MACS)*len(codes)+int(COSTS[codes[codes>=0]].sum()))/len(codes)/1e9,
            'stage2_fraction':float((codes>=0).mean())}
    save_json(out/'latency.json',{'summary':summary,'records':records,'gpu':torch.cuda.get_device_name(),
        'measurement':'synchronized wall time, batch 1, includes narrow/gating/Top-K/recycling/selected wide; excludes decoding/preprocessing/input H2D',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_sha256':hashlib.sha256((ROOT/'baseline/target80_runtime.py').read_bytes()).hexdigest()})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
