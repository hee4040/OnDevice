"""Package completed Colab summaries as a small, reviewable Git report.

Input is the downloaded layerwise-drop-v1 directory (or a bundle containing its
accuracy/, latency-gpu/ and figures/ subdirectories). Images, tokens, weights and
per-batch tensors are never copied.
"""
import argparse
import json
from pathlib import Path
import shutil


def read(directory, filename):
    return json.loads((directory / filename).read_text(encoding='utf-8'))


def fmt(value, places=3):
    return '—' if value is None else f'{value:.{places}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    accuracy = read(args.input / 'accuracy', 'summary.json')
    latency = read(args.input / 'latency-gpu', 'summary.json')
    environment = read(args.input / 'accuracy', 'environment.json')
    if environment['arguments']['samples'] != 50000 or any(row['samples'] != 50000 for row in accuracy['rows']):
        raise ValueError('Only a completed 50,000-image evaluation may be exported as the full report')
    if len(accuracy['rows']) != len(environment['configs']):
        raise ValueError('Missing configurations')
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ('accuracy', 'latency-gpu'):
        target = args.output / mode
        target.mkdir(exist_ok=True)
        for filename in ('summary.json', 'summary.csv', 'environment.json'):
            shutil.copy2(args.input / mode / filename, target / filename)
    if (args.input / 'figures').is_dir():
        target = args.output / 'figures'
        target.mkdir(exist_ok=True)
        for name in ('pareto.png', 'schedules.png'):
            shutil.copy2(args.input / 'figures' / name, target / name)
    if (args.input / 'model-check.json').exists():
        shutil.copy2(args.input / 'model-check.json', args.output / 'model-check.json')
    lines = [
        '# Layer-wise Drop: Colab 측정 결과', '',
        f"실행 장치: {environment['device']}. ImageNet-1K validation 50,000장, τ={accuracy['threshold']}, 공식 EMA, 추가 학습 없음.",
        f"정밀도: {environment['precision']}. PyTorch {environment['torch']}.",
        f"평가 코드 commit: `{environment['project']['commit']}`.",
        f"공식 ThinkingViT commit: `{environment['official']['commit']}`.", '',
        '## 무엇을 실험했나', '',
        'ThinkingViT는 먼저 작은 3H 모델을 실행하고, 예측 entropy가 τ 이상인 어려운 이미지만 큰 6H 모델로 보낸다. '
        '이번 실험은 routing과 가중치는 고정한 채 6H의 12개 Transformer Block에서 이미지 패치 토큰을 언제 제거할지 비교했다.', '',
        '3H 마지막 Block의 head 평균 CLS attention으로 196개 패치의 중요도 순위를 한 번 정한다. '
        '각 6H Block 진입 전에 그 순위의 Top-K만 남기며, 제거한 토큰은 복원하지 않고 CLS 토큰은 항상 유지한다. '
        '학습이나 파인튜닝은 하지 않았다.', '',
        '| 설정 | Block 1–4 | Block 5–8 | Block 9–12 | 의미 |',
        '| --- | ---: | ---: | ---: | --- |',
        '| `full` | 100% | 100% | 100% | 제거하지 않는 기준 |',
        '| `uniform_r` | r | r | r | 처음부터 끝까지 같은 비율 |',
        '| `staged_r` | 100% | (100%+r)/2 | r | 네 Block마다 계단식 감소 |',
        '| `linear_r` | 100%에서 시작 | 선형 감소 | r에서 종료 | Block마다 조금씩 감소 |',
        '| `late_r` | 100% | 100% | r | 마지막 네 Block에서만 제거 |', '',
        '`r`은 25%, 50%, 75% 중 하나이며 원래 패치 196개에 대한 유지율이다. '
        '예를 들어 50%는 패치 98개와 CLS를 합친 99토큰이다. `uniform_n148`은 모든 Block에서 패치 148개와 CLS를 유지한다.', '',
        '같은 최종 유지율이라도 staged/linear/late는 앞 Block에서 더 많은 토큰을 계산한다. '
        '따라서 각 가변 스케줄은 전체 6H MACs가 가장 가까운 `matched_uniform`과 비교해야 한다. '
        '`uniform_gain_pp`는 이 동일 연산량 대조군 대비 paired 정확도 차이다.', '',
        '## 정확도와 연산량', '',
        '| 설정 | 전체 Top-1 (%) | Routed Top-1 (%) | 평균 GMACs | 가까운 고정 대조군 | 차이 (%p) | MACs 차이 (%) |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: |',
    ]
    for row in accuracy['rows']:
        lines.append(f"| {row['name']} | {fmt(row['top1'])} | {fmt(row['top1_routed'])} | {fmt(row['avg_gmacs'])} | "
                     f"{row['matched_uniform'] or '—'} | {fmt(row.get('uniform_gain_pp'))} | {fmt(row.get('uniform_cost_delta_percent'))} |")
    main_comparison = next(r for r in accuracy['rows'] if r['name'] == 'staged_50')
    low, high = main_comparison['uniform_gain_ci95_pp']
    staged_75 = next(r for r in accuracy['rows'] if r['name'] == 'staged_75')
    late_75 = next(r for r in accuracy['rows'] if r['name'] == 'late_75')
    s75_low, s75_high = staged_75['uniform_gain_ci95_pp']
    l75_low, l75_high = late_75['uniform_gain_ci95_pp']
    lines += ['', '사전에 정한 단계적 50% 설정의 가까운 고정 대조군 대비 paired 차이:',
              f"{fmt(main_comparison['uniform_gain_pp'])}%p; 정규근사 95% 구간 [{fmt(low)}, {fmt(high)}]%p.",
              '구간은 개별 고정 비교용이며 다중 비교나 최고 설정 선택 효과를 보정하지 않았다.', '',
              '## 결론', '',
              '- 25% 유지에서는 모든 가변 스케줄이 동일 연산량 uniform보다 유의하게 나빴다.',
              '- 50% 유지에서는 staged/linear/late 모두 동일 연산량 uniform과 통계적으로 구분되지 않았다.',
              f"- staged_75는 동일 연산량 대조군보다 {fmt(staged_75['uniform_gain_pp'])}%p 높았다 "
              f"(95% 구간 [{fmt(s75_low)}, {fmt(s75_high)}]%p).",
              f"- late_75는 동일 연산량 대조군보다 {fmt(late_75['uniform_gain_pp'])}%p 높았다 "
              f"(95% 구간 [{fmt(l75_low)}, {fmt(l75_high)}]%p).",
              '- 따라서 레이어별 제거가 항상 유리하지는 않으며, 이번 조건에서는 완만한 75% 유지에서만 작은 이점이 관찰됐다.', '',
              '## 실제 전체 추론 시간', '',
              '| 설정 | 이미지 수 | 평균 (ms) | p50 (ms) | p95 (ms) | 실제 6H 진입 비율 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in latency['rows']:
        lines.append(f"| {row['name']} | {row['samples']} | {fmt(row['mean_ms'])} | {fmt(row['p50_ms'])} | "
                     f"{fmt(row['p95_ms'])} | {fmt(row['routed_fraction'])} |")
    lines += ['', 'MACs는 Block별 토큰 수로 계산한 해석적 추정이며 정렬/gather·norm·softmax·분기 비용은 제외한다.',
              '실제 시간은 batch 1에서 3H·attention 수집(drop)·gate·필요한 6H·정렬/gather·동기화를 포함한다.',
              '이미지 읽기·전처리·전송은 시간에서 제외한다. Colab GPU 측정치이며 모바일 실측이 아니다.',
              '같은 validation에서 여러 스케줄을 비교했으므로 최고 점수를 독립 검증 성능으로 해석하지 않는다.', '',
              '![Accuracy and compute](figures/pareto.png)', '', '![Token schedules](figures/schedules.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
