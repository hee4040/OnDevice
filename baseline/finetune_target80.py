#!/usr/bin/env python3
"""One preregistered epoch at fixed 50% retention, with full optimizer checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from evaluate_drop import ROOT,add_paths,load_model,wide_logits,NARROW_P
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention,make_update_mask
from analyze_adaptive_drop import save_json


def losses(narrow,wide,teacher_narrow,teacher_wide,target,smoothing):
    kl=lambda s,t:F.kl_div(F.log_softmax(s.float(),1),F.log_softmax(t.float(),1),reduction='batchmean',log_target=True)
    return [kl(narrow,teacher_narrow),kl(wide,teacher_wide),
            F.cross_entropy(narrow.float(),target,label_smoothing=smoothing),F.cross_entropy(wide.float(),target,label_smoothing=smoothing)]

def save_checkpoint(out,student,optimizer,step,total,variant):
    state={'model':student.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'total_steps':total,'variant':variant}
    torch.save(state,out/'training_state.tmp');os.replace(out/'training_state.tmp',out/'training_state.pt')
    torch.save(student.state_dict(),out/'student.tmp');os.replace(out/'student.tmp',out/'student.pth')

def main():
    ap=argparse.ArgumentParser(description=__doc__);add_paths(ap)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--variant',choices=['kd50','ce_kd50'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after',type=int)
    args=ap.parse_args();out=args.output;protocol=json.loads((ROOT/'configs/target80_20261108.json').read_text());cfg=protocol['finetune']
    if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=protocol['student_sha256']:raise ValueError('Wrong initial student')
    out.mkdir(parents=True,exist_ok=args.resume)
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    student=load_model(args.repo.resolve(),args.checkpoint.resolve(),args.weights)
    teacher=load_model(args.repo.resolve(),args.checkpoint.resolve()).requires_grad_(False)
    from timm.data import create_transform,resolve_data_config
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    config=resolve_data_config({},model=student)
    dataset=ImageFolder(args.data_dir/'train',transform=create_transform(**config,is_training=True))
    if len(dataset)!=1281167:raise ValueError(f'Incomplete train dataset: {len(dataset)}')
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=args.workers,pin_memory=True,
                      drop_last=True,persistent_workers=True,prefetch_factor=2,
                      generator=torch.Generator().manual_seed(cfg['seed']))
    total=len(loader);optimizer=torch.optim.AdamW(student.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    step=0
    if args.resume:
        state=torch.load(out/'training_state.pt',map_location='cuda',weights_only=True)
        if state['variant']!=args.variant or state['total_steps']!=total:raise ValueError('Resume mismatch')
        student.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];del state
    else:
        save_json(out/'protocol.json',protocol)
        save_json(out/'arguments.json',{'variant':args.variant,'seed':cfg['seed'],'train_images':len(dataset),'total_steps':total,
            'batch_size':cfg['batch_size'],'weights':str(args.weights.resolve()),'teacher_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'student_sha256':protocol['student_sha256'],'train_transform':str(dataset.transform),'torch':str(torch.__version__),
            'gpu':torch.cuda.get_device_name(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'mode':'eval mode for model, gradients enabled; no dropout or drop-path in this architecture',
            'checkpoint_selection':'final epoch only; no validation during training'})
    start_step=step;start=time.perf_counter();running=np.zeros(4);count=0
    weights=cfg['variants'][args.variant]
    print(f'{args.variant}: {step}/{total} steps; output={out}',flush=True)
    for batch_index,(images,target) in enumerate(loader):
        if batch_index<start_step:continue
        if args.stop_after is not None and step>=args.stop_after:break
        images=images.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
        lr=cfg['lr']*min(1.,(step+1)/cfg['warmup_steps'])*.5*(1+math.cos(math.pi*step/total))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            with torch.no_grad():
                teacher.p=NARROW_P;tf=teacher.forward_features(images,NARROW_P);tn=teacher.forward_head(tf)
                tw=wide_logits(teacher,wide_input(teacher,images,tf))
            sf,sn,att=forward_narrow_with_attention(student,images)
            sw=wide_logits(student,wide_input(student,images,sf),make_update_mask(att.float(),.5,'topk'))
        terms=losses(sn,sw,tn,tw,target,cfg['ce_label_smoothing']);loss=sum(w*t for w,t in zip(weights,terms))
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(student.parameters(),cfg['clip_grad_norm'])
        if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
        optimizer.step();step+=1;running+=np.array([t.item() for t in terms]);count+=1
        if step in [1,10] or step%100==0 or step==total:
            elapsed=time.perf_counter()-start
            row={'step':step,'total_steps':total,'mean_losses':(running/count).tolist(),'lr':lr,
                 'images_per_second':(step-start_step)*cfg['batch_size']/elapsed,
                 'elapsed_seconds':elapsed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
            print(json.dumps(row),flush=True)
            with (out/'train_log.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            save_json(out/'status.json',{'status':'running',**row});running[:]=0;count=0
        if step%1000==0 or step==total:save_checkpoint(out,student,optimizer,step,total,args.variant)
    if step%1000 and step!=total:save_checkpoint(out,student,optimizer,step,total,args.variant)
    status={'status':'complete' if step==total else 'stopped_at_requested_step','step':step,'total_steps':total,
            'weights_sha256':hashlib.sha256((out/'student.pth').read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start}
    save_json(out/'status.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':main()
