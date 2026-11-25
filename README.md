# Adaptive Token Allocation for Efficient ThinkingViT via Token Pruning and Knowledge Distillation

Token Pruning과 Knowledge Distillation을 활용한 ThinkingViT의 적응형 추론 효율 최적화

## 1. 연구 주제와 제안 방법

ThinkingViT는 3H 예측의 엔트로피에 따라 3H에서 종료하거나 6H로 확장한다. 기본 6H 경로는 패치 토큰을 모두 처리한다. 제안 방법은 3H 마지막 블록의 CLS Attention으로 패치 토큰을 정렬하고, 선택한 Top-K와 CLS 토큰만 6H에 전달한다. 공식 ThinkingViT EMA를 Teacher로 사용한 Knowledge Distillation(KD)으로 압축 경로의 정확도를 회복한다.

중심 기여는 **3H 종료 조건과 6H Token Retention Ratio의 공동 조정**이다. KD 가중치를 고정한 뒤 3H 상위 10개 클래스 확률을 재정규화한 엔트로피로 이미지별 계산을 배분한다. 아래는 사전 고정한 Balanced 정책이다.

| 3H 엔트로피 | 추론 경로 |
| --- | --- |
| 0.8 미만 | 3H 종료 |
| 0.8 이상 1.25 미만 | 6H, 패치 25% 유지 |
| 1.25 이상 | 6H, 패치 50% 유지 |

Accuracy 정책은 경계 0.75 / 1.15 / 1.75와 유지율 25% / 50% / 75%를 사용한다.

**평가 기준**은 전달받은 **RTX 3070 학습 KD 가중치**(`kd_student.pth`, SHA-256 `37e9a36a3a33edc1baf7ef06c93ddb970cc3a4fafea92a413acc24eea2d628ae`)와 **RTX 4090 측정**이다. 공식 EMA는 원본과 학습 전 대조군에 사용한다. Top-1 및 평균 GMACs는 ImageNet-1K validation 50,000장, 통합 GPU 지연시간은 배치 1, 고정된 1,000장, 3회 반복의 평균 지연시간 중앙값이다. 시간에는 게이트 판단과 GPU 동기화를 포함하며, 전처리·입력 전송·최종 argmax는 제외한다. PyTorch 2.5.1+cu124를 사용했다. 기본 정밀도는 FP32이며 TF32를 허용한다. MAC은 곱셈-누산 한 번으로 계산하며 정렬·메모리 이동은 지연시간에 반영된다. [4090 검증 로그](results/rtx4090-20261122T105147Z/validate-real-fp32.json)

## 2. Main Experiment

### 2.1 Token Drop + KD

RTX 4090에서 τ=1로 평가했다. 이 표의 GMACs는 레이어별 평가기의 동일 MAC 규약으로 산출했다.

| 방법 | Top-1 | 평균 GMACs |
| --- | ---: | ---: |
| 공식 EMA, 전체 토큰 | 80.304% | 2.9149 |
| 공식 EMA, 25% 유지 | 73.814% | 1.6705 |
| 공식 EMA, 50% 유지 | 78.610% | 2.0693 |
| 공식 EMA, 75% 유지 | 79.952% | 2.4841 |
| 3070 KD, 25% 유지 | 78.388% | 1.6721 |
| **3070 KD, 50% 유지** | **79.672%** | **2.0725** |
| 3070 KD, 75% 유지 | 79.980% | 2.4889 |

50% 유지에서 KD로 **+1.062%p**를 회복했다. 공식 전체 토큰 기준 대비 연산은 약 **29%** 줄었고 정확도 차이는 **−0.632%p**였다. [학습 전](results/rtx4090-20261122T105147Z/drop-before-kd.log) · [3070 KD](results/rtx4090-20261122T105147Z/drop-after-kd.log) · [공식 EMA GMACs](results/rtx4090-20261122T105147Z/layerwise-accuracy/summary.json) · [3070 KD GMACs](results/rtx4090-3070-kd/layerwise-accuracy/summary.json)

### 2.2 Adaptive Token Allocation

같은 3070 가중치로 RTX 4090에서 3H, 6H 25%, 6H 50% 예측을 생성했다. 고정 정책은 그대로 평가했고, 예산 탐색 정책은 시드 42로 나눈 25,000장에서 선택한 뒤 나머지 25,000장에서 확인했다.

| 정책 | 전체 50K Top-1 | 평균 GMACs | 홀드아웃 25K Top-1 | 홀드아웃 GMACs |
| --- | ---: | ---: | ---: | ---: |
| Fixed Drop 50%, τ=1 | 79.672% | 2.0805 | 79.288% | 2.0851 |
| 사전 고정 Balanced | 80.022% | 2.0714 | — | — |
| Joint, 기준 예산에서 탐색 | **80.090%** | **2.0792** | **79.652%** | **2.0830** |
| 사전 고정 Accuracy, 75% 경로 포함 | 80.440% | 2.2651 | — | — |

기준 예산 Joint는 경계 0.75 / 1.30을 선택했다. Fixed 대비 전체 **+0.418%p**, 홀드아웃 **+0.364%p**였으며 양쪽 모두 GMACs가 소폭 낮았다. 홀드아웃의 이미지별 짝지은 부트스트랩 95% 구간은 +0.160~+0.564%p였다(20,000회, seed 42). 정책 탐색에는 추가 모델 학습이 없다. 기존 Target80의 **79.652% → 80.076%**는 이전 저장 예측을 재계산한 결과다. 이 표의 주 결과는 3070 가중치를 RTX 4090에서 직접 평가한 값이다. [새 정책 원자료](results/rtx4090-3070-kd/allocation.json) · [Target80 이전 기록](results/rtx4090-20261122T105147Z/target80-replay-retry.log)

### 2.3 최종 통합 모델

| 모델 | 정밀도 | Top-1 | 평균 GMACs | GPU 지연시간 |
| --- | --- | ---: | ---: | ---: |
| 공식 EMA Native | FP32 | 80.304% | 2.9438 | 5.835 ms |
| 공식 EMA + 고정 폭·임베딩 재사용·CUDA Graph | FP32 | 80.304% | 2.9230 | 1.222 ms |
| 3070 KD Economy + CUDA Graph | FP32 | 78.388% | 1.6801 | 1.069 ms |
| 3070 KD Balanced + CUDA Graph | FP32 | 80.022% | 2.0714 | 1.156 ms |
| **3070 KD Accuracy + CUDA Graph** | **FP32** | **80.440%** | **2.2651** | **1.188 ms** |
| 3070 KD Fixed 50% + CUDA Graph | FP32 | 79.672% | 2.0805 | 1.115 ms |
| 3070 KD 전체 토큰 + CUDA Graph | FP32 | 80.040% | 2.9294 | 1.218 ms |
| 공식 EMA + 고정 폭·임베딩 재사용·CUDA Graph | FP16 | 80.300% | 2.9236 | 0.672 ms |
| 3070 KD Economy + CUDA Graph | FP16 | 78.420% | 1.6802 | 0.629 ms |
| 3070 KD Balanced + CUDA Graph | FP16 | 80.034% | 2.0717 | 0.670 ms |
| 3070 KD Accuracy + CUDA Graph | FP16 | 80.438% | 2.2652 | 0.691 ms |
| 3070 KD Fixed 50% + CUDA Graph | FP16 | 79.662% | 2.0808 | 0.640 ms |
| 3070 KD 전체 토큰 + CUDA Graph | FP16 | 80.042% | 2.9298 | 0.669 ms |

FP32 Accuracy는 공식 Native 대비 **+0.136%p, GMACs 23.05% 감소, 지연시간 79.6% 감소**였다. 동일한 FP32 CUDA Graph 실행기에서 공식 EMA와 비교하면 GMACs는 **22.5%**, 지연시간은 **2.8%** 감소했다. 같은 FP16 CUDA Graph에서 Accuracy는 공식 EMA보다 +0.138%p 높았으며 지연시간은 0.019 ms 길었다. Balanced는 80.034% / 0.670 ms였다. 공식 EMA의 Native → 최적화 실행 차이는 **5.835 → 1.222 ms**로, 전체 시간 개선에서 실행 최적화가 차지하는 부분을 보여준다. [FP32 정확도](results/rtx4090-20261122T105147Z/accuracy-fp32.json) · [FP32 GPU 시간](results/rtx4090-20261122T105147Z/latency-integrated-vs-original-fp32.json) · [KD FP16 GPU 시간](results/rtx4090-20261122T105147Z/latency-graph-fp16.json) · [공식 FP16 정확도](results/rtx4090-3070-kd/official-fp16-accuracy.json) · [공식 FP16 GPU 시간](results/rtx4090-3070-kd/official-fp16-latency.json)

아래 p95·직렬 처리량·peak CUDA allocated는 각 행의 평균 지연시간이 중앙값인 반복에서 읽었다. 메모리는 PyTorch가 기록한 프로세스 할당량이다.

| 모델 | p95 | 직렬 처리량 | Peak CUDA allocated |
| --- | ---: | ---: | ---: |
| 공식 EMA Native FP32 | 9.887 ms | 171.4 img/s | 672.8 MiB |
| 공식 EMA Graph FP32 | 1.952 ms | 818.6 img/s | 739.8 MiB |
| KD Accuracy Graph FP32 | 1.710 ms | 841.8 img/s | 739.8 MiB |
| 공식 EMA Graph FP16 | 1.008 ms | 1,488.0 img/s | 399.1 MiB |
| KD Accuracy Graph FP16 | 1.016 ms | 1,446.6 img/s | 399.1 MiB |

## 3. Ablation Study

코드는 [실험별 파일·원본 커밋·재현 명령](experiments/README.md)에 정리했다.

### 3.1 Token 처리 방식

6H 진입부터 패치 50%만 갱신하거나 유지했다. 두 열 모두 RTX 4090에서 평가했다.

| 방법 | 공식 EMA Top-1 | 3070 KD Top-1 |
| --- | ---: | ---: |
| Selective Refinement (Attention + MLP) | 57.772% | 57.758% |
| Full Attention + Selective MLP | 62.256% | 63.964% |
| **Token Drop** | **78.610%** | **79.672%** |

선택적 갱신은 갱신하지 않은 토큰을 시퀀스에 남긴다. 실제 제거가 두 가중치 모두에서 더 높은 정확도를 얻었다. [공식 EMA H1](results/ablation-worktrees/h1/results/20261122T114914.442466Z-h1-diagnosis/summary.json) · [3070 KD H1](results/rtx4090-3070-kd/h1-kd-summary.json)

### 3.2 Token Selection Strategy

같은 토큰 제거 구조, τ=1, 50% 유지에서 선택 기준만 바꿨다.

| 선택 기준 | 공식 EMA Top-1 | 3070 KD Top-1 |
| --- | ---: | ---: |
| **마지막 CLS Attention Top-K** | **78.610%** | **79.672%** |
| Random-K | 77.400% | 78.152% |
| CLS Attention Bottom-K | 69.220% | 69.964% |
| 낮은 3H Token Norm 우선 | — | 74.980% |
| Global CLS | — | 79.642% |

마지막 CLS Attention Top-K가 이 비교에서 가장 높았다. Global CLS는 같은 3070 KD를 사용한 3.6절의 Evo 평가에서 가져왔다. [H1 선택 진단](results/rtx4090-3070-kd/h1-kd-summary.json) · [Global CLS 원자료](results/rtx4090-3070-kd/evo/summary.json)

### 3.3 Fine-tuning Strategy

| 학습 방법 | 50% 유지 Top-1 |
| --- | ---: |
| 공식 EMA, 학습 전 | 78.610% |
| **제공된 3070 KD** | **79.672%** |
| 4090에서 새로 학습한 기본 KD | 79.596% |
| 4090 KD + CE | 79.338% |
| 4090 KD + Full KL | 79.650% |
| 4090 KD + Feature MSE | 79.618% |
| 4090 Partial FT (P+LN+LoRA) | 79.496% |

학습 손실과 학습 파라미터의 비교에는 각 방법으로 학습한 가중치가 필요하다. 4090 학습 결과는 학습 방식 비교와 재현성에 사용했다. 같은 통합 Accuracy 정책의 FP32 결과는 **3070 가중치 80.440% / 2.2651 GMACs / 1.188 ms**, **4090 새 가중치 80.234% / 2.2654 GMACs / 1.186 ms**였다. 정확도 차이는 **0.206%p**였으며 추론은 두 경우 모두 4090에서 측정했다. [학습 원자료와 25%·75% 결과](reports/RTX4090_REPRODUCTION.md)

### 3.4 Fixed vs Adaptive Token Allocation

모든 행은 같은 3070 가중치와 4090에서 생성한 예측을 사용한다. Gate-only는 50% 유지율을 고정한다. Ratio-only는 τ=1을 고정하고 25%와 50%를 구분한다. Joint는 두 경계를 함께 탐색한다. 캘리브레이션 기준 예산은 2.0760 GMACs이며, 0.025 간격 그리드에서 예산 이하의 가장 가까운 후보 주변 0.25% 범위 내 정확도를 기준으로 선택했다.

| 정책 | 종료 / 비율 경계 | 전체 Top-1 | 전체 GMACs | 홀드아웃 Top-1 | 홀드아웃 GMACs | GPU 지연시간 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 50% | 1.00 / — | 79.672% | 2.0805 | 79.288% | 2.0851 | 1.117 ms |
| Gate-only | 1.00 / — | 79.672% | 2.0805 | 79.288% | 2.0851 | 1.117 ms |
| Ratio-only | 1.00 / 1.025 | 79.670% | 2.0691 | 79.288% | 2.0738 | 1.115 ms |
| Joint | 0.75 / 1.30 | **80.090%** | 2.0792 | **79.652%** | 2.0830 | 1.173 ms |

같은 예산에서 Gate-only는 Fixed와 동일한 정책을 선택했다. Ratio-only는 캘리브레이션 그리드상 기준 예산에 가장 가까운 비퇴화 정책이며, 홀드아웃에서 Joint보다 0.0092 GMACs 적게 사용했다. 같은 FP32 CUDA Graph 벤치마크의 Joint 지연시간은 Fixed보다 약 0.057 ms 길었다. [Allocation 지연시간](results/rtx4090-3070-kd/allocation-latency.json)

| 캘리브레이션 예산 | Gate-only 홀드아웃 Top-1 / GMACs | Ratio-only 홀드아웃 Top-1 / GMACs | Joint 홀드아웃 Top-1 / GMACs |
| ---: | ---: | ---: | ---: |
| 1.9722 (−5%) | 78.724% / 1.9704 | 79.116% / 1.9766 | **79.300%** / 1.9774 |
| 2.0760 (기준) | 79.288% / 2.0851 | 79.288% / 2.0738 | **79.652%** / 2.0830 |
| 2.1797 (+5%) | 79.572% / 2.1825 | 79.288% / 2.0738 | **79.888%** / 2.1900 |

비용 제약은 캘리브레이션에서 적용했다. 홀드아웃에서는 이미지별 엔트로피 분포가 달라 예산을 소폭 넘을 수 있다. Ratio-only는 τ=1과 최대 50% 유지 때문에 +5% 예산까지 사용하지 못한다. [세 예산 원자료](results/rtx4090-3070-kd/allocation.json)

### 3.5 Layer-wise Token Drop

같은 3070 KD 가중치로 RTX 4090에서 레이어별 21개 정책을 평가했다. 시간은 배치 1, 1,000장, 이미지별 3회 반복의 Eager 평균이다. 표의 Uniform 172·180 토큰은 Staged·Late와 비슷한 연산량의 대조군이다.

| 정책 | Top-1 | 평균 GMACs | GPU 지연시간 |
| --- | ---: | ---: | ---: |
| 전체 토큰 | 80.040% | 2.9213 | 6.139 ms |
| Uniform 75% | 79.980% | 2.4889 | 5.925 ms |
| Staged 75% | **80.054%** | 2.7059 | 5.958 ms |
| Linear 75% | **80.054%** | 2.7039 | 6.119 ms |
| Late 75% | 80.024% | 2.7772 | 5.931 ms |
| Uniform 172 토큰 | 80.002% | 2.7075 | 5.922 ms |
| Uniform 180 토큰 | 80.018% | 2.7784 | 5.910 ms |

비슷한 연산량에서 Staged 75%는 Uniform 172 토큰보다 +0.052%p, Late 75%는 Uniform 180 토큰보다 +0.006%p였다. 25% 최종 유지의 Linear는 79.284% / 2.2858 GMACs였고, 비슷한 비용의 Uniform 123 토큰은 79.772% / 2.2830이었다. [3070 KD 전체 결과](results/rtx4090-3070-kd/layerwise-accuracy/summary.json) · [3070 KD 지연시간](results/rtx4090-3070-kd/layerwise-latency/summary.json) · [공식 EMA 대조군](results/rtx4090-20261122T105147Z/layerwise-accuracy/summary.json)

### 3.6 Token Information Preservation — Evo-ViT

전달받은 3070 KD 가중치로 모든 방법을 RTX 4090에서 평가했다. 6H 토큰 유지율은 50%, τ=1이다. 시간은 이 실험의 Eager 실행기에서 배치 1, 1,000장 평균이다.

| 방법 | Top-1 | GPU 지연시간 |
| --- | ---: | ---: |
| Last CLS Token Drop | **79.672%** | **6.554 ms** |
| Global CLS Token Drop | 79.642% | 7.643 ms |
| Evo Layer | 78.594% | 9.193 ms |
| Evo Stage | 78.682% | 8.792 ms |

같은 가중치에서 Global CLS, Evo 모두 Last CLS보다 시간이 길고 정확도가 낮았다. 4090에서 해당 구조로 별도 학습했을 때는 Global CLS 79.690%, Global CLS에서 시작한 Evo Layer 79.666%까지 회복됐다. [3070 KD Evo 원자료](results/rtx4090-3070-kd/evo/summary.json) · [학습 변형별 결과](reports/RTX4090_REPRODUCTION.md)

### 3.7 GPU Execution Optimization

공식 EMA에서 배치 1 FP32 Eager의 6H 구간 중앙값은 전체 토큰 5.348 ms, 25% 유지 5.458 ms, 50% 유지 5.460 ms, 75% 유지 5.503 ms였다. 이 구간의 벤치마크는 메인 표의 종단 간 지연시간과 측정 범위가 다르다. 별도 H1 최적화 스크립트에서는 CUDA Graph 캡처와 `torch.compile`이 오류로 끝났다. 통합 실행기로 조건을 맞춘 비교는 아래에 있다. [H1 최적화 로그](results/rtx4090-20261122T105147Z/hardware-optimization.log)

같은 3070 KD와 배치 1, 1,000장, 3회 반복에서 통합 FP32 Eager의 전체 토큰은 1.954 ms, Fixed 50%는 1.985 ms, Balanced는 2.099 ms, Accuracy는 2.143 ms였다. 같은 정책의 FP32 CUDA Graph 시간은 각각 1.218, 1.115, 1.156, 1.188 ms였다. [Eager 원자료](results/rtx4090-3070-kd/kd-eager-latency.json) · [Graph 원자료](results/rtx4090-20261122T105147Z/latency-integrated-vs-original-fp32.json)

Fixed 50%의 통합 FP32 Eager에서 6H 진입 시 패치 임베딩을 다시 계산하면 2.003 ms, 3H에서 계산한 임베딩을 재사용하면 1.985 ms였다. 측정 차이는 0.018 ms다. [임베딩 재계산 원자료](results/rtx4090-3070-kd/kd-reembed-latency.json)

TF32 허용 상태의 `torch.compile`은 Eager logits 일치 검사에서 실패했다. TF32를 모두 끈 뒤 같은 Fixed 50% 경로로 재측정하면 다음과 같다. 컴파일과 Graph 준비 시간은 지연시간에서 제외했다. 세 경로 모두 실제 이미지 32장의 logits·분기 검사를 통과했다.

| 통합 실행기, 엄격 FP32 | GPU 지연시간 |
| --- | ---: |
| Eager | 1.938 ms |
| `torch.compile` | 1.485 ms |
| CUDA Graph | **1.270 ms** |

같은 조건의 Eager 대비 `torch.compile`은 23.4%, CUDA Graph는 34.5% 빨랐다. [TF32 허용 실패 로그](results/rtx4090-3070-kd/kd-compiled-latency.log) · [엄격 FP32 Eager](results/rtx4090-3070-kd/kd-eager-strict-latency.json) · [엄격 FP32 컴파일](results/rtx4090-3070-kd/kd-compiled-strict-latency.json) · [엄격 FP32 Graph](results/rtx4090-3070-kd/kd-graph-strict-latency.json)

## 4. 논문 구성

| 섹션 | 내용 |
| --- | --- |
| Introduction / Related Work | ThinkingViT의 추가 추론 비용, 중첩 ViT, Token Pruning, KD, Adaptive Inference |
| Method | CLS Attention Token Drop → Drop-aware KD → Entropy 기반 공동 배분 |
| Main Results | Token Drop + KD → Adaptive Allocation → RTX 4090 통합 성능 |
| Ablation | 토큰 처리 → 토큰 선택 → 학습 방식 → 동일 예산 배분 → 레이어별 제거 → Evo-ViT → GPU 실행 |
| Generalization | ImageNet-V2, 추가 학습 seed 및 다른 GPU |
| Conclusion | 같은 계산량에서의 정확도와 실행 최적화를 통한 지연시간 개선 |

## 5. 추가 검증 상태

### 5.1 ImageNet-V2 일반화

원본 ImageNet-1K에서 결정한 정책을 ImageNet-V2 matched-frequency 10,000장에 그대로 적용했다. RTX 4090 FP32에서 평가했으며, KD는 메인 결과와 같은 3070 가중치다.

| 정책 | Top-1 | 평균 GMACs |
| --- | ---: | ---: |
| 공식 EMA 전체 토큰, τ=1 | 69.14% | 3.4511 |
| KD Fixed 50% / Gate-only | 68.30% | 2.3353 |
| KD Ratio-only | 68.31% | 2.3230 |
| KD Joint | 68.47% | 2.3124 |
| KD Balanced | 68.47% | 2.3091 |
| KD Accuracy | 68.93% | 2.5859 |

Joint는 Fixed보다 **+0.17%p**였고 계산량은 약 **0.98%** 적었다. 정확도 차이의 이미지별 짝지은 부트스트랩 95% 구간은 **−0.15~+0.49%p**였다(20,000회, seed 42). 공식 EMA 대비 Accuracy는 −0.21%p, GMACs 약 25.1% 감소였다. [데이터 원본](https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-matched-frequency.tar.gz) · [공식 EMA 원자료](results/rtx4090-3070-kd/v2/official.json) · [3070 KD 원자료](results/rtx4090-3070-kd/v2/kd.json)

### 5.2 검증 범위

Gate-only·Ratio-only·Joint의 세 예산 비교, 동일 정밀도·실행기의 통합 비교, 4090에서의 ImageNet-V2 검증을 완료했다. 복수 학습 seed에서 정확도 변동을 정량화하거나 다른 모델·GPU로 확장하는 실험은 이번 범위에서 남아 있다. 별도 Target80 체크포인트의 GPU 재실행 여부는 이 브랜치에서 검증하지 않았다.

## 6. 핵심 연구 기여

3070에서 학습한 하나의 KD 가중치를 고정하고 4090에서 평가했을 때, 3H 종료와 6H 토큰 유지율을 공동 조정한 정책은 거의 같은 연산량에서 Fixed Drop 대비 **전체 +0.418%p**, **홀드아웃 +0.364%p**를 얻었다. FP32 최종 통합 모델은 **80.440% / 2.2651 GMACs / 1.188 ms**를 기록했다. 공식 Native·공식 Graph·통합 Graph의 세 측정으로 실행 최적화와 KD·배분 정책 적용 후의 변화량을 확인했다.

## 7. 설치와 입력 준비

```bash
bash scripts/bootstrap.sh
```

공식 ThinkingViT revision은 `9254c4950d3159937ad682592d58a2be69708c42`로 고정한다.
기존 checkout이 다른 revision이면 스크립트는 그 checkout을 자동으로 바꾸지 않고 종료한다.
공식 EMA는 `checkpoints/thinkingvit_3h6h_ema.pth`, 학습된 KD는 `checkpoints/kd_student.pth`에 둔다.
가중치는 Git에 포함하지 않는다. 이 컴퓨터의 `checkpoints/kd_student.pth`는 전달받은 RTX 3070 학습 가중치를 가리킨다.

ImageNet 사용 승인과 인증을 완료한 뒤 필요한 split을 준비한다.

```bash
# 평가만 할 때
.venv/bin/python scripts/prepare_imagenet.py --splits validation
# KD 학습까지 할 때 추가
.venv/bin/python scripts/prepare_imagenet.py --splits train
```

기본 데이터 위치는 `data/imagenet1k/{train,validation}/<class>/...`다. 기존 데이터가 준비돼 있으면 재다운로드할 필요가 없다.
RTX 4090 실행의 해시와 GPU 정보는 [FP32 원자료](results/rtx4090-20261122T105147Z/accuracy-fp32.json)에 기록했다.

## 8. 검증과 평가 재현

새 출력 디렉터리를 사용한다. 기존 결과 파일은 덮어쓰지 않는다. 정확도 평가의 `.pt` 저장을 위해 디렉터리도 먼저 만든다.

```bash
RUN_DIR="results/recheck-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

# 원본 / 고정 폭 / Graph 출력 검증: 실제 이미지 32장 포함
.venv/bin/python -m efficient_vit validate --real-images --output "$RUN_DIR/validate-real-fp32.json"

# 주 대조군: main의 공식 EMA, Drop·KD 없음
.venv/bin/python -m efficient_vit evaluate --official --runtime native --policies full --output "$RUN_DIR/accuracy-original-fp32.json"
.venv/bin/python -m efficient_vit benchmark --official --runtime native --policies full --output "$RUN_DIR/latency-original-fp32.json"

# 같은 원본 가중치·정책에 실행 최적화만 적용
.venv/bin/python -m efficient_vit evaluate --official --policies full --output "$RUN_DIR/accuracy-official-optimized-fp32.json"
.venv/bin/python -m efficient_vit benchmark --official --runtime graph --policies full --output "$RUN_DIR/latency-official-graph-fp32.json"

# 통합 모드 전체: 정확도·분기별 평균 GMACs·실제 시간
.venv/bin/python -m efficient_vit evaluate --output "$RUN_DIR/accuracy-fp32.json"
.venv/bin/python -m efficient_vit benchmark --runtime graph --output "$RUN_DIR/latency-integrated-vs-original-fp32.json"

# 코드·데이터 준비 테스트
.venv/bin/python -m unittest discover -s tests

# 원자료에서 공유용 요약 JSON과 reports/RESULTS.md 생성
.venv/bin/python scripts/report.py --source "$RUN_DIR"
```

`--samples 128`은 실행 점검용이다. 5만 장 결과로 보고하지 않는다.
시간 측정은 다른 GPU compute 작업이 없는 상태에서 실행한다. Graph는 **batch 1 전용**이며 전체 정확도는 eager 배치 경로로 측정한다.
RTX 4090에서 단위 테스트 12개와 실제 이미지 32장의 원본·고정 폭·Graph 출력 검증을 통과했다.

공동 배분의 세 예산 비교는 같은 3070 가중치로 재현한다. JSON에는 분할별 결과, NPZ에는 이미지별 엔트로피와 경로별 정답 여부가 기록된다.

```bash
.venv/bin/python -m experiments.allocation.evaluate --output "$RUN_DIR/allocation.json"
```

FP16은 같은 `RUN_DIR`에서 별도로 검증한다.

```bash
.venv/bin/python -m efficient_vit validate --precision fp16 --real-images --output "$RUN_DIR/validate-fp16.json"
.venv/bin/python -m efficient_vit evaluate --precision fp16 --output "$RUN_DIR/accuracy-fp16.json"
.venv/bin/python -m efficient_vit benchmark --runtime graph --precision fp16 --output "$RUN_DIR/latency-graph-fp16.json"
.venv/bin/python scripts/report.py --source "$RUN_DIR"
```

## 9. KD 학습 재현

동일 KD 가중치가 없으면 아래 recipe로 새로 학습할 수 있다. 새 학습의 정확도가 위 체크포인트와 bit 단위로 같다는 보장은 없다.

| 항목 | 설정 |
| --- | --- |
| Teacher | 공식 EMA 3H·전체 토큰 6H, 동결 |
| 학습량 | ImageNet train 1,281,167장, 3에폭 |
| batch | 유효 128 / micro 32, gradient accumulation |
| 손실 | 0.5 KL(3H)+0.5 KL(Drop 6H), T=1 |
| 유지율 | batch마다 25% 또는 50% 선택 |
| optimizer | AdamW, lr 5e-5, weight decay 0 |
| schedule | warmup 500, cosine decay |
| 기타 | BF16, gradient clip 1, seed 42 |

```bash
.venv/bin/python -m efficient_vit.train --output results/kd-train
# 중단한 경우
.venv/bin/python -m efficient_vit.train --output results/kd-train --resume results/kd-train/resume.pth
# 새 가중치 평가
.venv/bin/python -m efficient_vit evaluate --weights results/kd-train/student.pth --output results/new-kd-accuracy.json
```

재개 시 optimizer·schedule·RNG·batch 위치를 복원한다. worker augmentation의 bit 단위 재현은 보장하지 않는다.
정책 경계는 학습 후 적용한다. 학습 모듈은 가중치를 공유하는 원래 native 모델을 사용하며,
폭별 가중치 복사·Graph 캡처는 추론 단계에서만 적용한다.

## 10. Python 추론

```python
from PIL import Image
from efficient_vit import load_native, specialize, Engine

native = load_native(
    "checkpoints/thinkingvit_3h6h_ema.pth",
    "checkpoints/kd_student.pth",
)
# load_native가 공식 custom timm의 import 경로를 설정한 다음 가져온다.
from timm.data import create_transform, resolve_data_config

transform = create_transform(
    **resolve_data_config({}, model=native, use_test_size=True),
    is_training=False,
)
engine = Engine(specialize(native), runtime="graph")
images = transform(Image.open("example.jpg").convert("RGB")).unsqueeze(0).cuda()
logits, path, entropy = engine.predict(images, policy="balanced")
prediction = logits.argmax(1).item()
```

기본 입력은 CUDA FP32 `[1, 3, 224, 224]`다. FP16을 쓰려면 `specialize(..., dtype=torch.float16)`과
입력의 dtype을 함께 바꾼다. `path=-1`은 3H 종료이며 나머지는 해당 정책의 `ratios` 인덱스다.
반환 logits와 entropy는 다음 Graph 호출 후에도 유효하도록 복사한다.

## 11. 파일과 결과 근거

작업 브랜치 `repro/rtx4090-efficient-kd`는 `hee4040/OnDevice`의 `integration/efficient-kd` 커밋 `7b398be`에서 생성했다.

| 경로 | 내용 |
| --- | --- |
| [`efficient_vit/policy.py`](efficient_vit/policy.py) | 고정 정책·entropy·MACs 산식 |
| [`efficient_vit/model.py`](efficient_vit/model.py) | 원본 로딩·고정 폭 변환·embedding 재사용 |
| [`efficient_vit/runtime.py`](efficient_vit/runtime.py) | eager/Graph 조건부 추론 |
| [`efficient_vit/verify.py`](efficient_vit/verify.py) | main 원본 경로와 출력 비교 |
| [`efficient_vit/__main__.py`](efficient_vit/__main__.py) | 검증·평가·시간 측정 CLI |
| [`efficient_vit/train.py`](efficient_vit/train.py) | 보존한 전체 KD 학습 recipe |
| [`scripts/report.py`](scripts/report.py) | 공유용 요약 생성 |
| [`reports/RESULTS.md`](reports/RESULTS.md) | 기준 브랜치의 이전 통합 결과 |
| [`reports/original-main-protocol.json`](reports/original-main-protocol.json) | main revision·체크포인트 해시·원본 설정 |
| [`reports/original_equivalence.json`](reports/original_equivalence.json) | 공식 원본에 실행 최적화만 적용한 5만 장 예측·분기 일치 |
| [`reports/native_equivalence.json`](reports/native_equivalence.json) | KD 원본/통합 대조군의 5만 장 일치 |
| [`reports/implementation_sha256.json`](reports/implementation_sha256.json) | 검증한 구현 해시 |

이번 RTX 4090 원자료는 로컬 `results/rtx4090-20261122T105147Z/`와 `results/rtx4090-3070-kd/`에 있다. `reports/RESULTS.md`와 기존 추적 JSON은 기준 브랜치의 이전 실행 기록이다.
결과와 체크포인트 경로는 Git 제외 대상이므로 새 클론에서는 평가 명령으로 생성한다. 추가 평가의 로컬 실행 기록은 [종료 상태](results/rtx4090-3070-kd/status.tsv)와 각 로그에 있다.
이전 실험과 선택 근거는 `experiment/drop-ft`, `codex/target80-results`, `experiment/allocation-rule`,
`codex/layerwise-drop-results`, `experiment/h1-diagnosis` 브랜치에 보존돼 있다.
