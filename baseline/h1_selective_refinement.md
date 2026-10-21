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
- ImageNet-1K 전체 실험: Hugging Face 인증 부재로 실행 대기

## 전체 validation 결과

2026-10-21 현재 작업 폴더에는 ImageNet-1K validation 데이터와 Hugging Face 접근 인증이 없다.
`scripts/prepare_imagenet.py --splits validation`을 실제 실행했으며, ImageNet 이용 조건 동의 및
`huggingface-cli login`이 필요하다는 오류로 데이터 요청 전에 중단됐다. 데이터 없이 기존 집계값이나
예시 이미지로 H1의 결과를 추정하지 않는다.

| 선택 정책 | 패치 갱신 비율 | 전체 Top-1 | Correction Retention | Wide-correct Retention | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 실행 대기 | — | — | — | — | ImageNet validation 필요 |

따라서 현재 단계에서는 구현 동작과 공식 6H 동등성을 검증했지만 H1을 채택하거나 기각할 수 없다.

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
