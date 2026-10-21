# H1 선택적 토큰 갱신 실험

## 연구 질문

ThinkingViT에서 3H → 6H 확장으로 얻는 정확도 개선이 전체 패치 토큰을 다시 갱신하지 않아도
유지되는지 검증한다.

구체적인 H1 판정 질문은 다음과 같다.

> 3H 마지막 블록의 CLS Attention Top-K로 선택한 25%, 50%, 75% 패치만 6H에서 갱신했을 때,
> 전체 6H가 교정한 이미지의 정답을 얼마나 유지하며 같은 갱신 비율의 Random-K 및 Bottom-K보다
> 우수한가?

## 구현한 실험 의미론

- 기존 Entropy threshold `1.0`으로 6H 진입 이미지를 선택한다.
- 선택 대상은 196개 패치 토큰이며 CLS 토큰은 항상 갱신한다.
- 선택 점수는 3H 마지막 블록에서 head 평균한 CLS Attention이다.
- 선택 마스크는 이미지마다 한 번 만들고 12개 6H 블록에서 고정한다.
- 각 6H Attention에서 선택 토큰은 Query로 갱신되며 전체 현재 토큰을 Key/Value 문맥으로 사용한다.
- 비선택 토큰은 각 블록 진입 전 상태를 그대로 유지한다.
- 6H 입력은 공식 Token Recycling과 동일하게 원본 임베딩과 투영된 3H 최종 특징을 융합한다.

현재 구현은 모든 토큰에 대해 block을 계산한 뒤 선택된 상태만 반영하는 **dense-mask accuracy
oracle**이다. 따라서 이 결과로 FLOPs 또는 latency 감소를 주장하지 않는다. H1/H2가 성립한 뒤
Query projection, Attention, output projection과 MLP를 선택 토큰에만 실행하는 sparse 구현이 별도로
필요하다.

## 실험 조건

| 항목 | 설정 |
| --- | --- |
| 모델 | ThinkingViT DeiT 3H → 6H |
| 가중치 | 공식 일반 EMA export |
| 데이터 | ImageNet-1K validation 50,000장 |
| 입력 | 224×224, 공식 evaluation transform |
| Entropy threshold | 1.0 |
| 갱신 비율 | 0%, 25%, 50%, 75%, 100% |
| 패치 수 | 0, 49, 98, 147, 196개; CLS 별도 포함 |
| 선택 정책 | CLS Attention Top-K, Bottom-K, Random-K |
| Random 반복 | seed 0, 1, 2, 3, 4 |
| 정밀도 | FP32, AMP off, TF32 허용 |

100% 갱신 결과는 공식 전체 6H 결과와 logits 허용 오차 및 Top-1이 일치해야 한다. 이 검증이
실패하면 실험을 중단한다.

## 평가 지표

- 전체 early-exit Top-1/Top-5
- 6H 진입 이미지 Top-1
- Correction Retention: 기존 6H가 `3H 오답 → 6H 정답`으로 교정한 이미지 중 선택적 갱신도
  정답인 비율
- Wide-correct Retention: 기존 6H가 맞힌 routed 이미지 중 선택적 갱신도 정답인 비율
- Regression Recovery: 기존 6H가 `3H 정답 → 6H 오답`으로 악화한 이미지 중 선택적 갱신이
  정답을 회복한 비율
- 전체 6H와의 routed Top-1 예측 불일치 수

## 구현 및 검증 상태

- 구현 브랜치: `experiment/h1-selective-token-refinement`
- 공식 ThinkingViT 코드는 수정하지 않았다.
- 공식 ThinkingViT commit: `9254c4950d3159937ad682592d58a2be69708c42`
- 정확도 oracle: `baseline/selective_refinement.py`
- 전체 평가기: `baseline/evaluate_selective.py`
- 단위 테스트: `tests/test_selective_refinement.py`
- 전체 단위 테스트: 28개 통과
- Python 문법 및 patch 형식 검사: 통과
- GPU 통합 검사: NVIDIA GeForce RTX 3070 Ti Laptop GPU, batch 2 합성 입력에서 통과
- 합성 입력 100% 갱신과 공식 6H logits 최대 절댓값 차이: `0.0`
- 합성 입력 100% 갱신과 공식 6H Top-1 일치: `true`
- CLS Attention shape 및 행 합 검사: `(2, 197)`, 행 합 최대 오차 `0.0`
- 갱신 토큰 수 검사: 0/25/50/75/100%에서 CLS 포함 1/50/99/148/197개
- 25% 부분 갱신 logits 유한성 검사: 통과
- ImageNet-1K 전체 실험: 50,000장 완료
- 원본 결과: `results/20261021T142511.222335Z-selective-refinement/`
- 추적용 요약: `baseline/reports/2026-10-21/selective_refinement.json`

## 전체 validation 결과

2026-10-21 ImageNet-1K validation 50,000장 전체를 batch 32로 평가했다. Entropy threshold 1.0에서
18,007장이 6H로 진입했다. 100% 갱신 경로는 공식 전체 6H 경로와 첫 routed batch의 logits가 정확히
일치했고, 전체 early-exit Top-1은 80.312%였다. 기존 batch 128 측정의 80.304% 및 공식 batch 512
측정의 80.310%와의 작은 차이는 기존 문서에 기록된 batch 구성·GPU 수치 연산 영향 범위다.

| 선택 정책 | 패치 갱신 비율 | 전체 Top-1 | Routed Top-1 | Correction Retention | Wide-correct Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLS만 갱신 | 0% | 57.858% | 0.405% | 0.254% (11/4,333) | 0.257% (29/11,300) |
| CLS Attention Top-K | 25% | 57.800% | 0.244% | 0.208% (9/4,333) | 0.186% (21/11,300) |
| CLS Attention Bottom-K | 25% | 57.830% | 0.328% | 0.277% (12/4,333) | 0.230% (26/11,300) |
| Random-K, 5 seeds 평균 | 25% | 57.810 ± 0.007% | 0.271 ± 0.019% | 0.245 ± 0.035% | 0.200 ± 0.019% |
| CLS Attention Top-K | 50% | 57.768% | 0.155% | 0.138% (6/4,333) | 0.124% (14/11,300) |
| CLS Attention Bottom-K | 50% | 57.814% | 0.283% | 0.231% (10/4,333) | 0.212% (24/11,300) |
| Random-K, 5 seeds 평균 | 50% | 57.779 ± 0.006% | 0.187 ± 0.018% | 0.143 ± 0.034% | 0.143 ± 0.007% |
| CLS Attention Top-K | 75% | 57.784% | 0.200% | 0.162% (7/4,333) | 0.204% (23/11,300) |
| CLS Attention Bottom-K | 75% | 57.778% | 0.183% | 0.208% (9/4,333) | 0.142% (16/11,300) |
| Random-K, 5 seeds 평균 | 75% | 57.770 ± 0.007% | 0.162 ± 0.019% | 0.097 ± 0.044% | 0.140 ± 0.029% |
| 전체 갱신 | 100% | 80.312% | 62.753% | 100% (4,333/4,333) | 100% (11,300/11,300) |

Random-K의 `±`는 5개 seed의 표본 표준편차다. Correction/Wide-correct Retention의 `±`는 퍼센트포인트
단위다.

## 판정

이번 실험에서 H1은 **현재 구현한 형태로는 지지되지 않는다.** 75%의 패치를 갱신해도 routed Top-1이
0.200%에 불과했고, 전체 6H가 교정한 4,333장 중 7장만 유지했다. 25%와 50%도 같은 붕괴를 보였다.
Top-K는 같은 비율의 Bottom-K나 Random-K보다 일관되게 우수하지 않았으므로 마지막 3H 블록의 CLS
Attention이 “추가 width 계산이 필요한 토큰”을 잘 식별한다는 증거도 얻지 못했다.

이 판정 범위는 다음 조건으로 제한한다.

- 기존 가중치를 고정한 inference-only 실험
- 비선택 토큰을 12개 6H 블록 모두에서 완전히 동결
- 한 이미지에서 같은 선택 마스크를 모든 6H 블록에 사용
- 별도 fine-tuning 또는 lightweight update 경로 없음

따라서 “어떤 선택적 토큰 계산도 불가능하다”는 결론은 아니다. 결과는 pretrained ThinkingViT에 단순
고정 마스크를 drop-in하는 방식이 실패한다는 것을 보여준다. 선택 토큰과 동결 토큰이 다음 블록의
Key/Value에서 서로 다른 갱신 단계의 표현으로 섞이면서 분포가 깨졌을 가능성이 있지만, 이 원인은
추가 진단이 필요하다.

H2도 현재 구조에서는 기각된다. 가장 좋은 부분 갱신 전체 Top-1은 57.858%로 100% 갱신의 80.312%보다
22.454%p 낮다. H3의 실제 sparse latency 구현은 정확도 전제가 성립하지 않아 아직 진행하지 않는다.

다음 실험은 원인을 분리하는 순서가 적절하다.

1. Attention은 전체 토큰을 갱신하고 MLP만 선택적으로 생략한다.
2. 초기 wide block은 전체 갱신하고 후반 block에서만 선택적으로 동결한다.
3. 비선택 토큰에 identity가 아닌 저비용 update 경로를 둔다.
4. 위 방법 중 정확도가 회복되는 구조에 한해서 fine-tuning을 수행한다.

## 재현 명령

```bash
# ImageNet 접근 승인을 받은 Hugging Face 계정으로 한 번 로그인
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/huggingface-cli login

# validation 50,000장 준비
.venv/bin/python scripts/prepare_imagenet.py --splits validation

# H1 전체 실험
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate_selective.py \
  --ratios 0 .25 .5 .75 1 \
  --policies topk bottomk random \
  --random-seeds 0 1 2 3 4 \
  --batch-size 32
```

결과 원본은 `results/<UTC 시각>-selective-refinement/` 아래의 `summary.json`, `summary.csv`,
`environment.json`, `sample_indices.json`에 저장된다.
