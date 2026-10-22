# H1 후속 실험: 전체 Attention + 선택적 MLP 갱신

## 연구 질문

이전 실험은 비선택 토큰의 Attention과 MLP 갱신을 모두 동결했고, 75% 패치를 갱신해도 전체
Top-1이 57.784%로 붕괴했다. 이번 실험은 원인을 분리하기 위해 다음 질문을 검증한다.

> 모든 토큰이 6H Attention을 통해 계속 정보를 주고받게 하고, MLP residual만 일부 토큰에서
> 생략하면 전체 6H의 정확도 개선을 유지할 수 있는가?

## 구현 의미론

각 6H Transformer block에서 다음 순서로 계산한다.

1. 모든 197개 토큰에 Attention residual을 적용한다.
2. Attention 이후 상태를 입력으로 모든 토큰의 MLP 후보를 계산한다.
3. CLS와 선택된 패치에만 MLP residual을 반영한다.
4. 비선택 패치는 Attention 이후 상태를 유지한 채 다음 block으로 전달한다.

선택 마스크는 3H 마지막 block의 head-평균 CLS Attention으로 한 번 만들고 12개 6H block에서
고정한다. CLS는 항상 MLP 갱신 대상이다. Token Recycling과 Entropy gate는 기존 ThinkingViT와
같다.

현재 코드는 MLP를 모든 토큰에 계산한 뒤 residual만 마스킹하는 **dense-mask accuracy oracle**이다.
따라서 이번 결과는 정확도 가설만 판정하며 FLOPs 또는 latency 감소를 측정하거나 주장하지 않는다.
실제 절감을 검증하려면 선택 토큰을 gather하여 MLP를 실행한 뒤 scatter하는 별도 sparse 구현이 필요하다.

## 실험 조건

| 항목 | 설정 |
| --- | --- |
| 모델 | ThinkingViT DeiT 3H → 6H |
| 가중치 | 공식 일반 EMA export |
| 데이터 | ImageNet-1K validation 50,000장 |
| 입력 | 224×224, 공식 evaluation transform |
| Entropy threshold | 1.0 |
| MLP 패치 갱신 비율 | 0%, 25%, 50%, 75%, 100% |
| Attention 갱신 비율 | 항상 100% (CLS 포함 197개) |
| 선택 정책 | CLS Attention Top-K, Bottom-K, Random-K |
| Random 반복 | seed 0, 1, 2, 3, 4 |
| 정밀도 | FP32, AMP off, TF32 허용 |

100% MLP 갱신 경로는 공식 전체 6H 경로와 첫 routed batch의 logits가 정확히 일치했으며 최대
절댓값 오차는 `0.0`이었다.

## 전체 validation 결과

2026-10-23에 batch 32로 50,000장 전체를 평가했다. threshold 1.0에서 18,007장이 6H로
진입했다. 기존 전체 6H가 교정한 이미지는 4,333장, routed 이미지 중 전체 6H가 맞힌 이미지는
11,300장이다.

| 선택 정책 | MLP 패치 갱신 | 전체 Top-1 | Routed Top-1 | Correction Retention | Wide-correct Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLS만 MLP 갱신 | 0% | 57.812% | 0.278% | 0.231% (10/4,333) | 0.372% (42/11,300) |
| CLS Attention Top-K | 25% | 59.264% | 4.309% | 3.485% (151/4,333) | 5.681% (642/11,300) |
| CLS Attention Bottom-K | 25% | 58.036% | 0.900% | 0.808% (35/4,333) | 1.133% (128/11,300) |
| Random-K, 5 seeds 평균 | 25% | 58.578 ± 0.039% | 2.406 ± 0.107% | 1.842 ± 0.136% | 3.112 ± 0.172% |
| CLS Attention Top-K | 50% | 62.248% | 12.595% | 10.755% (466/4,333) | 16.982% (1,919/11,300) |
| CLS Attention Bottom-K | 50% | 58.820% | 3.077% | 2.400% (104/4,333) | 3.929% (444/11,300) |
| Random-K, 5 seeds 평균 | 50% | 60.853 ± 0.063% | 8.722 ± 0.174% | 6.716 ± 0.142% | 11.669 ± 0.258% |
| CLS Attention Top-K | 75% | 66.716% | 25.001% | 23.102% (1,001/4,333) | 34.628% (3,913/11,300) |
| CLS Attention Bottom-K | 75% | 60.674% | 8.225% | 6.808% (295/4,333) | 10.796% (1,220/11,300) |
| Random-K, 5 seeds 평균 | 75% | 63.336 ± 0.025% | 15.616 ± 0.069% | 13.358 ± 0.361% | 21.214 ± 0.114% |
| 전체 MLP 갱신 | 100% | 80.312% | 62.753% | 100% (4,333/4,333) | 100% (11,300/11,300) |

Random-K의 `±`는 5개 seed의 표본 표준편차이며 retention의 단위는 퍼센트포인트다.

## 이전 실험과 비교

| Top-K 비율 | Attention+MLP 모두 선택 갱신 | 전체 Attention+선택 MLP | 개선 | 100% 대비 손실 |
| --- | ---: | ---: | ---: | ---: |
| 25% | 57.800% | 59.264% | +1.464%p | -21.048%p |
| 50% | 57.768% | 62.248% | +4.480%p | -18.064%p |
| 75% | 57.784% | 66.716% | +8.932%p | -13.596%p |

전체 Attention은 선택 비율이 높을수록 이전 방식보다 분명히 개선됐다. 따라서 비선택 토큰의
Attention까지 동결한 것이 이전 붕괴를 더 심하게 만든 요인이라는 증거는 얻었다. 그러나 Attention을
모두 갱신해도 75% MLP Top-K가 전체 6H 교정의 23.102%만 유지했으므로, 문제를 해결했다고 볼 수 없다.

Top-K는 25%, 50%, 75% 모두 Random-K와 Bottom-K보다 높았다. 즉 CLS Attention은 이번 구조에서는
무작위보다 유용한 순위를 제공한다. 다만 절대 정확도가 낮아 현재 선택 기준만으로 실용적인
정확도-연산량 절충을 만들지는 못했다.

## 판정과 다음 단계

이 실험의 가설은 **현재 inference-only 고정 마스크 조건에서는 지지되지 않는다.** MLP residual을
25%만 생략한 75% Top-K도 전체 Top-1이 13.596%p 하락했다. pretrained 모델은 매 block에서 모든
토큰이 Attention뿐 아니라 MLP 변환까지 받는 분포에 강하게 의존하는 것으로 보인다.

다음 실험은 정확도 붕괴 위치를 더 좁히는 것이 적절하다.

1. 앞쪽 wide block은 전체 MLP 갱신하고 뒤쪽 block에서만 선택적으로 생략한다.
2. block별 MLP 생략 민감도를 한 block씩 측정해 안전한 block을 찾는다.
3. 비선택 토큰에 저비용 adapter 또는 축소 MLP를 적용한다.
4. 정확도가 회복되는 구조에 한해 fine-tuning과 실제 sparse gather/scatter를 구현한다.

## 구현 및 재현

- 브랜치: `experiment/h1-attention-full-mlp-selective`
- 공식 ThinkingViT 코드는 수정하지 않았다.
- 공식 ThinkingViT commit: `9254c4950d3159937ad682592d58a2be69708c42`
- 구현: `baseline/mlp_selective_refinement.py`
- 평가기: `baseline/evaluate_mlp_selective.py`
- 테스트: `tests/test_mlp_selective_refinement.py`
- 전체 단위 테스트: 29개 통과
- 원본 결과: `results/20261022T175859.803228Z-mlp-selective-refinement/`
- 추적용 요약: `baseline/reports/2026-10-23/mlp_selective_refinement.json`

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate_mlp_selective.py \
  --ratios 0 .25 .5 .75 1 \
  --policies topk bottomk random \
  --random-seeds 0 1 2 3 4 \
  --batch-size 32
```
