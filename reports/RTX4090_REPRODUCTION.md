# RTX 4090 ThinkingViT 재현

이 문서는 11월 23일 UTC 실험 및 4090 재학습 기록이다. 전달받은 3070 가중치의 추가 4090 평가와 논문 순서의 최신 결과는 [README](../README.md)에 정리했다.

갱신 UTC 2026-11-23T18:43:17+00:00
통합 커밋 `7b398be40cee8c790f0e4bd645a29f6065247ec9`
원자료 및 실행 로그 `results/rtx4090-20261122T105147Z/`

KD 학생 가중치 SHA-256은 기존 통합 실험의 `37e9a36a3a33edc1baf7ef06c93ddb970cc3a4fafea92a413acc24eea2d628ae`와 일치한다.
Top-1 및 GMACs는 validation 50,000장. 지연시간은 batch 1, 1,000장, 3회 반복 기준이다.

| 모델 | Top-1 | 평균 GMACs | GPU 지연시간 (ms) |
| --- | ---: | ---: | ---: |
| 공식 EMA Native FP32 | 80.304% | 2.9438 | 5.835 |
| 공식 EMA CUDA Graph FP32 | 80.304% | 2.9230 | 1.222 |
| KD economy FP32 | 78.388% | 1.6801 | 1.069 |
| KD economy FP16 | 78.420% | 1.6802 | 0.629 |
| KD balanced FP32 | 80.022% | 2.0714 | 1.156 |
| KD balanced FP16 | 80.034% | 2.0717 | 0.670 |
| KD accuracy FP32 | 80.440% | 2.2651 | 1.188 |
| KD accuracy FP16 | 80.438% | 2.2652 | 0.691 |
| KD kd50 FP32 | 79.672% | 2.0805 | 1.115 |
| KD kd50 FP16 | 79.662% | 2.0808 | 0.640 |
| KD full FP32 | 80.040% | 2.9294 | 1.218 |
| KD full FP16 | 80.042% | 2.9298 | 0.669 |

## 실행 상태

| 실행 | 종료 코드 |
| --- | ---: |
| `unit-tests` | 0 |
| `validate-synthetic-fp32` | 0 |
| `validate-real-fp32` | 0 |
| `accuracy-original-fp32` | 0 |
| `accuracy-official-optimized-fp32` | 0 |
| `accuracy-fp32` | 0 |
| `validate-fp16` | 0 |
| `accuracy-fp16` | 0 |
| `latency-original-fp32` | 0 |
| `latency-official-graph-fp32` | 0 |
| `latency-integrated-vs-original-fp32` | 0 |
| `latency-graph-fp16` | 0 |
| `target80-restore` | 0 |
| `target80-replay` | 1 |
| `target80-replay-retry` | 0 |
| `h1-selection` | 0 |
| `drop-before-kd` | 0 |
| `drop-after-kd` | 0 |
| `layerwise-accuracy` | 0 |
| `layerwise-latency` | 0 |
| `hardware-optimization` | 1 |
| `imagenet-train` | 0 |
| `drop-ft-official` | 0 |
| `drop-ft-train` | 0 |
| `partial-ft-train` | 1 |
| `evolution-train` | 0 |
| `accuracy-retrained-kd-fp32` | 0 |
| `latency-retrained-kd-fp32` | 0 |
| `partial-ft-train-retry` | 0 |
| `evolution-warm-start` | 0 |

## 별도 입력이 필요한 실험

- Target80의 GPU 재현과 Allocation Rule의 ImageNet-V2 비교에는 별도 KD 가중치가 필요하다. 기존 수치 자료 재계산은 원본 저장 자료를 사용한다.
- Drop FT 손실별·Partial FT·Global CLS/Evo는 ImageNet train 준비 뒤 공식 EMA에서 새로 학습한다. 기존 통합 가중치의 결과와 새 학습의 결과를 구분한다.
- Gate-only·Ratio-only·Joint의 독립적인 동일 예산 비교와 통합 모델의 ImageNet-V2 평가는 추가 검증 항목이다.

## RTX 4090에서 새로 학습한 Drop FT

공식 EMA에서 시작해 ImageNet train 1,281,167장으로 학습했다. 아래는 validation 50,000장, FP32, τ=1 결과다.

| 학습 | 25% 유지 | 50% 유지 | 75% 유지 |
| --- | ---: | ---: | ---: |
| KD | 78.450% | 79.596% | 79.914% |
| KD + CE | 78.282% | 79.338% | 79.648% |
| KD + Full KL | 78.264% | 79.650% | 80.042% |
| KD + Feature MSE | 78.040% | 79.618% | 79.928% |
| 원본 탐색이 선택한 손실 조합 | 77.990% | 79.692% | 80.064% |

## 새 KD 가중치의 통합 실행기 평가

| 정책 | Top-1 | 평균 GMACs | GPU 지연시간 (ms) |
| --- | ---: | ---: | ---: |
| economy | 78.450% | 1.6801 | 1.067 |
| balanced | 79.932% | 2.0723 | 1.155 |
| accuracy | 80.234% | 2.2654 | 1.186 |
| kd50 | 79.596% | 2.0806 | 1.113 |
| full | 80.008% | 2.9295 | 1.216 |

## Partial FT 및 Evo

Partial FT의 첫 시도는 학습 후 산출물을 비교하는 단계에서 기존 원본 결과 경로가 없어 중단됐다. 해당 위치에 이번 동일 실행기의 공식 EMA 평가 결과를 연결하고, 원본 코드로 재개했다.

| 실험 | Drop 50% Top-1 (τ=1) |
| --- | ---: |
| Partial FT p | 79.326% |
| Partial FT p_ln | 79.342% |
| Partial FT p_ln_lora | 79.496% |
| evo_schedule / evo_layer | 79.662% |
| evo_schedule / evo_stage | 79.616% |
| existing_kd / last_cls | 79.596% |
| existing_kd / global_cls | 79.626% |
| existing_kd / evo_layer | 78.522% |
| existing_kd / evo_stage | 78.610% |
| global_cls / last_cls | 79.624% |
| global_cls / global_cls | 79.690% |
| global_to_evo / evo_layer | 79.666% |
| global_to_evo / evo_stage | 79.660% |
