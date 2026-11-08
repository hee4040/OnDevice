"""Actual conditional execution for locked joint exit/retention policies."""
import math
import torch
from selective_refinement import forward_narrow_with_attention,make_update_mask
from evaluate_h1_diagnosis import wide_input
from evaluate_drop import wide_logits
from target80_gate_search import RATIOS,number

def torch_codes(entropy,policy):
    e=entropy.double()
    if policy['kind']=='fixed':codes=torch.full_like(e,RATIOS.index(policy['ratio']),dtype=torch.long)
    else:codes=torch.where(e<number(policy['t1']),0,torch.where(e<number(policy['t2']),1,2)).long()
    return torch.where(e<number(policy['gate']),-1,codes)

def forward_target80(model,images,policy,narrow_result=None):
    if model.training:raise ValueError('Evaluation mode required')
    features,logits,attention=forward_narrow_with_attention(model,images) if narrow_result is None else narrow_result
    entropy=model.entropy(logits)[:,0];codes=torch_codes(entropy,policy);output=logits.clone()
    possible=[RATIOS.index(policy['ratio'])] if policy['kind']=='fixed' else [0,1]+([] if math.isinf(number(policy['t2'])) else [2])
    for code in possible:
        selected=(codes==code).nonzero(as_tuple=True)[0]
        if not len(selected):continue
        model.p=model._stage_ratio(6)
        x0=wide_input(model,images.index_select(0,selected),features.index_select(0,selected))
        mask=make_update_mask(attention.index_select(0,selected),RATIOS[code],'topk')
        output.index_copy_(0,selected,wide_logits(model,x0,mask))
    return output,codes,entropy
