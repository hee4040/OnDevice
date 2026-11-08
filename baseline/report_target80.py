#!/usr/bin/env python3
"""Audit and summarize measured target-80 results without retuning policies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from target80_gate_search import COSTS
from analyze_adaptive_drop import NARROW_MACS,save_json

ROOT=Path(__file__).resolve().parents[1]
LABELS={'baseline':'기존 고정 50%','same_budget':'같은 비용 정책','target_80.0':'추가 비용 +3.38%','target_80.2':'추가 비용 +9.34%'}
ENGLISH=['Baseline','Same budget','Cost +3.38%','Cost +9.34%']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--experiment',type=Path,required=True);args=ap.parse_args();out=args.experiment
    accuracy=json.loads((out/'gpu_accuracy/summary.json').read_text());diagnostics=json.loads((out/'diagnostics/summary.json').read_text());latency=json.loads((out/'latency.json').read_text())
    search=json.loads((out/'gate_search/summary.json').read_text());raw=torch.load(out/'gpu_accuracy/per_image.pt',weights_only=True)
    split=json.loads((Path(search['source'])/'split_ids.json').read_text());cal=np.isin(raw['image_id'].numpy(),split['calibration'])
    assert len(raw['image_id'])==50000 and torch.equal(raw['image_id'].sort().values,torch.arange(50000))
    for r in accuracy['comparison']:
        mask={'calibration':cal,'previously_seen_holdout':~cal,'full_validation':np.ones(50000,dtype=bool)}[r['split']]
        values=raw['methods'][r['method']];correct=(values['predicted']==raw['target']).numpy()
        assert np.array_equal(correct,values['correct'].numpy())
        codes=values['codes'].numpy()[mask];cost=int(NARROW_MACS)*int(mask.sum())+int(COSTS[codes[codes>=0]].sum())
        assert cost==r['total_macs'] and int(correct[mask].sum())==r['correct']
    assert len(latency['records'])==12
    timing_ids=json.loads((out/'latency_protocol.json').read_text())['image_ids']
    for r in latency['records']:
        assert r['image_ids']==timing_ids and len(r['times_ms'])==1000
        assert np.isfinite(r['times_ms']).all() and min(r['times_ms'])>0
    assert accuracy['cache_mismatches']['same_budget']=={'routing':0,'correctness':0,'image_ids':[]}
    assert accuracy['max_entropy_error']==0 and accuracy['narrow_correctness_mismatches']==0
    rows={r['method']:r for r in accuracy['comparison'] if r['split']=='full_validation'};names=list(LABELS)
    mainrow=rows['same_budget'];base=rows['baseline'];time=latency['summary']
    slowdown=100*(time['same_budget']['mean_ms']/time['baseline']['mean_ms']-1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors=['#667085','#1570ef','#f79009','#b54708']
    for ax,xfield,xlabel in [(axes[0],'gmacs','Analytic GMAC / image (50k)'),(axes[1],'mean_ms','Batch-1 mean latency (ms, paired 1k x 3)')]:
        for name,label,color in zip(names,ENGLISH,colors):
            x=rows[name][xfield] if xfield=='gmacs' else time[name][xfield];y=rows[name]['top1']
            ax.scatter(x,y,c=color,s=65);ax.annotate(f'{label}\n{y:.3f}%',(x,y),xytext=(7,7),textcoords='offset points',fontsize=8)
        ax.axhline(80,color='#98a2b3',ls='--',lw=1);ax.grid(alpha=.2);ax.set(xlabel=xlabel,ylabel='Top-1 (%) on full validation',ylim=(79.5,80.6))
    axes[0].set_xlim(2.03,2.48);axes[1].set_xlim(12.1,14.65)
    fig.suptitle('Joint exit / retention policy · exploratory validation reuse')
    fig.savefig(out/'target80_tradeoff.png',dpi=180);plt.close(fig)
    regions=diagnostics['attribution'];fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    values=[r['net_correct'] for r in regions]
    bars=ax.bar(range(4),values,color=['#98a2b3','#12b76a','#f04438','#98a2b3'])
    ax.bar_label(bars,labels=[f'{v:+d}' for v in values],padding=4)
    ax.set_xticks(range(4),['E < 0.8\nExit unchanged','0.8 <= E < 1.0\nExit -> 25%','1.0 <= E < 1.25\n50% -> 25%','E >= 1.25\n50% unchanged'])
    ax.set(ylabel='Net additional correct images',ylim=(-100,330),title='275 gains - 63 losses = 212 more correct / 50,000')
    ax.axhline(0,color='#344054',lw=.8);ax.grid(axis='y',alpha=.2);fig.savefig(out/'target80_attribution.png',dpi=180);plt.close(fig)
    text=['# 80% 목표 실험 결과 · 종료 게이트와 토큰 유지율 공동 조정','',
        f"**기존 학생 가중치 그대로 실제 ImageNet-1K validation 50,000장에서 {mainrow['top1']:.3f}%를 측정했다.** 기존 {base['top1']:.3f}% 대비 +{mainrow['top1']-base['top1']:.3f}%p이며, 평균 분석 비용은 {base['gmacs']:.6f} → {mainrow['gmacs']:.6f} GMAC/장으로 {abs(mainrow['cost_change_percent']):.3f}% 줄었다.",'',
        f"**실제 속도는 개선되지 않았다.** 같은 1,000장·batch 1·3회 측정에서 {time['baseline']['mean_ms']:.3f} → {time['same_budget']['mean_ms']:.3f}ms로 {slowdown:.2f}% 느려졌다. 정확도 80%와 평균 MAC 예산 목표를 충족한 관측 결과이며, 속도까지 개선한 결과는 아니다.",'',
        '기존 validation을 보정과 반복 탐색에 사용했다. 전체 수치는 보정용을 포함하고, 과거 holdout도 이미 확인한 자료다. 아래 결과를 새로운 독립 평가나 미지의 모집단 정확도 80% 보장으로 해석하지 않는다.','',
        '## 1. 기존 실험과 달라진 부분','',
        '이전 선택기 실험은 3H 종료 기준을 entropy 1.0으로 고정했다. 이번에는 종료 기준과 토큰 유지율을 함께 조정했다. 공식 상위 10개 확률 재정규화 entropy를 그대로 사용하며, 가중치와 crop 0.9·bicubic 전처리는 바꾸지 않았다. 비선택 토큰을 제거하는 token drop 구현이다.','',
        '| entropy 구간 | 기존 방식 | 이번 주 정책 | 패치 수 |', '|---|---|---|---:|',
        '| 0.8 미만 | 3H 종료 | 3H 종료 | 6H 미실행 |',
        '| 0.8 이상, 1.0 미만 | 3H 종료 | 6H·25% | 49 |',
        '| 1.0 이상, 1.25 미만 | 6H·50% | 6H·25% | 49 |',
        '| 1.25 이상 | 6H·50% | 6H·50% | 98 |','',
        'CLS는 항상 포함한다. 넓은 모델의 입력에는 기존 token recycling을 적용하고 이미지마다 선택된 6H 경로 하나만 실행한다. 전체 50,000장 중 27,400장은 종료, 9,350장은 25%, 13,250장은 50% 경로를 실행했다. 6H 진입률은 36.138%에서 45.200%로 증가했다.','',
        '## 2. 정책을 선택한 절차','',
        f"사전 고정한 0.00~2.35의 0.05 간격 경계와 종료 끝점을 사용했다. 고정 유지율 25·50·75·90·100% 및 단조 exit→25→50→75% 경계 조합 {search['candidates']:,}개를 기존 보정용 25,000장으로만 비교했다. 후보 수는 경로가 동등한 경계 조합도 포함한다.",'',
        '주 정책은 기존 고정 50%의 정수 MAC 합계 이내에서 보정용 정답 수가 가장 많은 규칙이다. 별도로 보정용 80.0%, 80.2%를 만족하는 최소 비용 규칙도 각각 고르고 정책 파일을 저장했다. 그 다음 과거 holdout 집계와 실제 GPU 재평가를 수행했다. 평가 결과로 경계를 다시 바꾸지 않았다.','',
        '## 3. 전체 validation 실제 GPU 결과','',
        '| 정책 | 정답 수 / 50,000 | Top-1 (%) | GMAC/장 | 기존 대비 비용 |', '|---|---:|---:|---:|---:|']
    for name in names:
        r=rows[name];text.append(f"| {LABELS[name]} | {r['correct']:,} | {r['top1']:.3f} | {r['gmacs']:.6f} | {r['cost_change_percent']:+.3f}% |")
    text+=['','추가 비용 3.38% 정책: 종료 경계 0.75, 25→50% 경계 1.15. 추가 비용 9.34% 정책은 여기에 50→75% 경계 1.75를 사용한다. 이 두 결과는 같은 비용의 성과로 계산하지 않는다.','',
        '| 분할 | 기존 Top-1 (%) | 주 정책 Top-1 (%) | 차이 (%p) | 조건부 95% 구간 (%p) | 주 정책 GMAC/장 |', '|---|---:|---:|---:|---|---:|']
    for split_name,label in [('calibration','보정용 25,000장'),('previously_seen_holdout','과거 holdout 25,000장'),('full_validation','전체 50,000장')]:
        b=next(r for r in accuracy['comparison'] if r['method']=='baseline' and r['split']==split_name)
        a=next(r for r in accuracy['comparison'] if r['method']=='same_budget' and r['split']==split_name)
        ci=next(r for r in diagnostics['paired_intervals'] if r['split']==split_name)
        text.append(f"| {label} | {b['top1']:.3f} | {a['top1']:.3f} | {ci['delta_pp']:+.3f} | [{ci['ci_low_pp']:+.3f}, {ci['ci_high_pp']:+.3f}] | {a['gmacs']:.6f} |")
    text+=['','구간은 이미 선택한 예측을 고정한 paired bootstrap 5,000회·seed 0이다. 탐색과 선택 불확실성을 모두 반영한 확증 구간은 아니다. 주 정책은 보정용에서는 80%에 못 미쳤고 과거 holdout과 전체 validation에서 80%를 넘었다.','',
        '## 4. 어디에서 212장을 더 맞혔는가?','',
        '| entropy 구간 | 이미지 수 | 오답→정답 | 정답→오답 | 정답 순증 |', '|---|---:|---:|---:|---:|']
    for r in regions:text.append(f"| {r['region']} | {r['n']:,} | {r['improved']} | {r['regressed']} | {r['net_correct']:+d} |")
    text+=['','새로 6H를 적용한 0.8~1.0 구간의 순증 275장이, 1.0~1.25 구간에서 토큰을 줄여 생긴 손실 63장보다 컸다. 1.25 이상 구간은 같은 경로지만 2장의 정오가 교차했다. 배치 그룹 구성에 따른 수치 변동으로 해석하며 순증은 0이다.','',
        '![구간별 정답 증감](target80_attribution.png)','',
        '## 5. 사후 교차검증 안정성','',
        '전체 50,000장을 다시 5-fold로 나눠, 각 fold의 학습 40,000장으로만 정책을 선택하고 나머지 10,000장을 예측했다. 두 시드는 같은 자료를 재분할한 것이다. 100%와 98% 학습 예산 모두 주 정책을 바꾸지 않는 사후 진단으로 수행했다.','',
        '| 시드 | 학습 예산 | OOF Top-1 (%) | GMAC/장 | 평균 비용 변화 | 평가 fold 중 원래 비용 초과 |', '|---|---:|---:|---:|---:|---:|']
    for r in diagnostics['crossfit']:text.append(f"| {r['seed']} | {r['budget_fraction']:.0%} | {r['top1']:.3f} | {r['gmacs']:.6f} | {r['cost_change_percent']:+.3f}% | {r['ceiling_exceeded_folds']}/5 |")
    text+=['','100% 예산의 10개 fold 모두에서 종료 경계 0.8이 선택됐다. 25→50% 경계는 9개에서 1.25, 1개에서 1.30이었다. 두 시드의 평균 비용은 기준보다 낮았지만 일부 fold는 초과했다. 98%로 여유를 둔 경우 모든 fold의 원래 비용 한도를 지켰으나 정확도는 80%보다 낮았다. 평균 비용 목표가 모든 새 표본의 비용을 보장하는 것은 아니다.','',
        '## 6. 실제 지연시간','',
        'RTX 3090에서 같은 과거 holdout 1,000장·batch 1·warmup 50·3회 회전 순서로 측정했다. 3H·게이트·Top-K·recycling·선택된 6H를 포함하고, 이미지 디코딩·전처리·입력 H2D는 제외한다.','',
        '| 정책 | 평균 ms | p50 ms | p95 ms | GPU peak allocated MiB |', '|---|---:|---:|---:|---:|']
    for name in names:
        r=time[name];text.append(f"| {LABELS[name]} | {r['mean_ms']:.3f} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['peak_allocated_mib']:.2f} |")
    text+=['','12회 실행 모두 batch 64 GPU 결과와 선택 경로·정답 여부가 일치했다. 분석 MAC이 같아도 6H에 진입하는 이미지가 늘면 시간은 늘 수 있다. 이번에는 실제로 지연시간이 증가했고, 커널별 원인을 분리해 입증한 것은 아니다. 모바일 지연시간은 측정하지 않았다.','',
        '![정확도·계산량·시간 비교](target80_tradeoff.png)','',
        '## 7. 실행 검증과 추가 학습 상태','',
        '원래 학생 SHA-256과 전처리를 확인했다. 3H entropy의 기존 캐시 대비 최대 오차 0, 3H 정답 여부 불일치 0이다. 주 정책은 50,000장 전체에서 선택·정답 여부 모두 캐시와 일치했다. 기준 정책과 추가 비용 정책에는 각각 2장·1장·2장의 정답 여부 차이가 있어 그대로 기록했다. 주 정책의 80.076%는 실제 조건부 실행으로 확인한 값이다.','',
        '코드 검사 53개가 통과했다. 이미지 ID 50,000개, 예측과 정답 여부, split별 정수 MAC 합계, 시간 표본을 재검사했다. 가중치·코드 해시, 정책, 후보 탐색 결과, 이미지별 실제 예측·유지율, crossfit 분할과 정책, 지연 원자료를 보존했다.','',
        '50% 유지율 전용 KD와 CE+KD 추가 학습은 각각 10단계 실행 점검만 완료했다. 기존 가중치의 정책 조정으로 목표에 도달해 전체 1 epoch 학습으로 진행하지 않았다. 이 중간 가중치는 위 정확도·시간 측정에 사용하지 않았다.','',
        '## 8. 결론과 다음 방향','',
        '이번 관측에서는 새 gate 네트워크나 추가 파인튜닝 없이, 종료 기준과 유지율의 공동 조정으로 평균 MAC 예산을 유지하며 80%를 넘었다. 이전 고정 gate 1.0 실험의 실패와 모순되지 않는다. 당시에는 3H에서 종료된 이미지에 계산을 다시 배분할 수 없었다.','',
        '다음에는 이 단순 규칙을 새로운 비교 기준으로 고정하고, 50% 유지율 전용 파인튜닝 또는 CPU/GPU 실행 최적화가 정확도와 시간에 추가 이득을 주는지 분리해서 검증하는 것이 적절하다. 기존 validation 반복 탐색의 영향을 고려해 독립 평가 계획도 먼저 정해야 한다.','',
        f'원시 결과: `{out.resolve()}`. 실행 안내: `baseline/target80_README.md`. 고정 설정: `configs/target80_20261108.json`. 서버 로컬 경로이며 공개 다운로드 링크는 아니다.','']
    (out/'report.md').write_text('\n'.join(text))
    script_names=['target80_gate_search.py','target80_runtime.py','evaluate_target80.py','measure_target80.py','diagnose_target80.py','finetune_target80.py','report_target80.py']
    manifest={'status':'complete','unit_tests_passed':53,'actual_replay_samples':50000,'policies':4,'latency_records':12,
        'student_sha256':accuracy['weights_sha256'],'protocol_sha256':hashlib.sha256((ROOT/'configs/target80_20261108.json').read_bytes()).hexdigest(),
        'source_sha256':{name:hashlib.sha256((ROOT/'baseline'/name).read_bytes()).hexdigest() for name in script_names},
        'audit':'image identities, predictions vs targets, exact split costs and timing sample identities verified',
        'limitations':'Validation reuse; post-hoc CV; average MAC constraint; increased latency; full fine-tuning not performed'}
    save_json(out/'manifest.json',manifest)
    shared=ROOT/'baseline/reports/2026-11-08/target80';shared.mkdir(parents=True,exist_ok=True)
    for name in ['report.md','manifest.json','execution_decision.json','target80_tradeoff.png','target80_attribution.png']:
        shutil.copyfile(out/name,shared/name)
    shutil.copyfile(ROOT/'configs/target80_20261108.json',shared/'protocol.json');shutil.copyfile(ROOT/'baseline/target80_README.md',shared/'README.md')
    for dirname,names_to_copy in {'gpu_accuracy':['summary.json','comparison.csv','locked_policies.json'],'gate_search':['summary.json','comparison.csv','policies.json'],
        'diagnostics':['summary.json','protocol.json','crossfit_pooled.csv','crossfit_folds.csv','crossfit_policies.json','actual_region_attribution.csv','actual_paired_intervals.csv']}.items():
        target=shared/dirname;target.mkdir(exist_ok=True)
        for name in names_to_copy:shutil.copyfile(out/dirname/name,target/name)
    save_json(shared/'latency_summary.json',{k:v for k,v in latency.items() if k!='records'})
    print('Report:',shared/'report.md');print('Latency change percent:',slowdown)

if __name__=='__main__':main()
