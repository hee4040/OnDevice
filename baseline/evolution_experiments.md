# Global CLS · Slow-fast 전체 실험 기록

갱신: 2026-11-15T20:26:12+09:00 (KST)

기존 실험 상태: **모든 학습·평가 완료**

이어받기 실험 상태: **이어받기 3에폭 학습·평가 완료**

## 실험 구성과 현재 판단

기존 전체 KD의 유지율 50%·entropy τ=1 정확도 79.672%를 기준으로 비교한다.
기존 KD 가중치에 선택 방식만 바꾼 사전 평가와, 변경 구조를 적용한 재학습 결과를 구분한다.
Global CLS 재학습 결과는 79.678%로 기준보다 +0.006%p(5만 장 중 3장) 높다.
단일 seed에서 사실상 같은 수준이며 뚜렷한 정확도 개선으로 주장하지 않는다.

| 실험 | 시작 가중치 | 학습 구조 | 추가 학습 | 누적 KD 학습 |
| --- | --- | --- | ---: | ---: |
| 기존 KD | 공식 EMA | 마지막 3H CLS로 고정 Top-K Drop | 3에폭 | 3에폭 |
| 선택 방식 사전 비교 | 완료된 기존 KD | 재학습 없이 네 추론 방식 적용 | 0 | 3에폭 |
| global_cls | 공식 EMA | Global CLS + 고정 Top-K Drop | 3에폭 | 3에폭 |
| evo_schedule | 공식 EMA | Global CLS + slow-fast + layer-to-stage | 3에폭 | 3에폭 |
| global_to_evo (추가) | 완료된 global_cls의 student.pth | Global CLS + slow-fast + layer-to-stage | 3에폭 | 6에폭 |

evo_schedule과 global_to_evo 모두 **Global CLS를 포함**한다. Global CLS 없이 slow-fast만 학습한 실험은 없다.
global_to_evo는 Global CLS 가중치만 학생에 복사한다. Teacher는 계속 공식 EMA 전체 토큰 모델이며,
optimizer·warmup·cosine·RNG·step은 새 학습으로 시작한다. 이전 optimizer 상태를 이어받지 않는다.
기존 evo_schedule 학습·평가를 먼저 끝내고 추가 실험을 순차 실행한다.
추가 실험은 총 6에폭이므로 초기 가중치 효과와 학습량 효과를 분리한 비교가 아니다.

## 구현한 방법

1. **Global CLS:** 새 학습 토큰을 추가하지 않는다. 3H의 첫 블록 CLS attention으로 초기화하고,
   이후 12블록에서 head 평균 attention을 `g = 0.5*g + 0.5*a`로 누적한다. 그 순위로 6H 토큰을 고른다.
2. **Slow-fast:** 6H에서 선택 패치는 직접 갱신하고, 비선택 패치는 global attention의 정규화 가중합으로
   대표 토큰 하나를 만든다. CLS·선택 패치·대표 토큰만 블록에 넣고 대표 토큰의 **출력−입력 잔차**를
   비선택 패치마다 더한다. 원래 위치를 보존해 이후 재선택이 가능하며 attention EMA는 선택 패치만 갱신한다.
3. **Layer-to-stage:** slow-fast 학습 첫 2에폭은 매 블록에서 재선택하고 마지막 1에폭은 4블록마다 재선택한다.
   각 블록의 특징·attention 갱신은 계속 수행한다. 기본 추론은 4블록 단위(evo_stage)다.
   같은 가중치에서 블록별 재선택(evo_layer)도 평가한다. 별도의 layer-only 학습 실험은 아니다.

3H는 197개 토큰을 유지하고, 6H 재활용 입력을 만든 뒤 압축한다. 대표 토큰도 기존 Drop과 같은 예산에 포함한다.
유지율 50%일 때 기존은 CLS 1+패치 98, slow-fast는 CLS 1+패치 97+대표 1로 둘 다 블록 입력이 99개다.
대표 생성·gather/scatter·점수 추출 비용은 추가되므로 토큰 수 일치가 실제 시간 일치를 뜻하지 않는다.
논문 Eq.4–7을 ThinkingViT에 적용한 변형이다. 공식 DeiT 코드는 앞 4블록을 dense로 처리하고 placeholder를
그대로 보존하는 변형을 사용한다. 이번 구현은 3H 점수로 6H 첫 블록부터 압축하고 논문 수식의 잔차를 전파한다.

## 공통 학습·평가 설정

| 항목 | 설정 |
| --- | --- |
| 데이터 | ImageNet-1K train 1,281,167장 / validation 50,000장 |
| 손실 | 0.5 KL(3H) + 0.5 KL(압축 6H), T=1; CE·Full·Feature 추가 없음 |
| 학습 | 전체 backbone, 3에폭·30,027 update, 유효 batch 128 / micro 32, seed 42 |
| optimizer | AdamW lr 5e-5, weight decay 0, warmup 500, cosine, gradient clip 1 |
| 유지율 | 학습 시 batch마다 25/50% 선택, 재학습 후 평가 25/50/75% |
| 정밀도 | BF16 학습 / FP32·TF32 허용 평가 |
| 평가 | seed-42 validation 순서, τ=1 고정, 마지막 checkpoint; validation으로 epoch 선택 안 함 |
| 모델·GPU | ThinkingViT 3H→6H, 224×224, RTX 3070 Ti Laptop 8GB |

유지율은 버린 비율이 아니라 남기는 패치 예산 비율이다. routed 정확도는 entropy≥1이면 압축 6H,
그 외에는 3H 정답을 사용한다. 전체 6H 단독 정확도와 이 지표는 다르다.

## 학습 진행

| 실험 | update | 완료율 | 상태 |
| --- | ---: | ---: | --- |
| global_cls | 30,027/30,027 | 100.0% | 학습 완료 |
| evo_schedule | 30,027/30,027 | 100.0% | 학습 완료 |
| global_to_evo | 30,027/30,027 | 100.0% | 학습 완료 |

## 재학습 전 사전 비교

완료된 기존 KD 가중치를 그대로 사용한 5만 장 평가다.

표본 50,000장 · 평가 완료 11-15 06:16:46 (KST)

| 추론 경로 | 해당 경로 단독 Top-1 | τ=1 최종 Top-1 |
| --- | ---: | ---: |
| narrow | 73.498% | 73.498% |
| wide | 81.092% | 80.040% |
| last_cls_50 | 80.688% | 79.672% |
| global_cls_50 | 80.652% | 79.642% |
| evo_layer_50 | 79.248% | 78.596% |
| evo_stage_50 | 79.404% | 78.682% |

## 구조 적용 후 재학습 결과

### global_cls

표본 50,000장 · 평가 완료 11-15 10:49:02 (KST)

| 추론 경로 | 해당 경로 단독 Top-1 | τ=1 최종 Top-1 |
| --- | ---: | ---: |
| narrow | 73.526% | 73.526% |
| wide | 81.104% | 79.988% |
| last_cls_25 | 79.006% | 78.280% |
| last_cls_50 | 80.586% | 79.564% |
| last_cls_75 | 81.066% | 79.934% |
| global_cls_25 | 79.216% | 78.448% |
| global_cls_50 | 80.744% | 79.678% |
| global_cls_75 | 81.096% | 79.984% |

### evo_schedule

표본 50,000장 · 평가 완료 11-15 15:35:49 (KST)

| 추론 경로 | 해당 경로 단독 Top-1 | τ=1 최종 Top-1 |
| --- | ---: | ---: |
| narrow | 73.496% | 73.496% |
| wide | 80.972% | 79.902% |
| evo_layer_25 | 78.872% | 78.216% |
| evo_layer_50 | 80.630% | 79.634% |
| evo_layer_75 | 80.992% | 79.894% |
| evo_stage_25 | 78.816% | 78.194% |
| evo_stage_50 | 80.552% | 79.572% |
| evo_stage_75 | 81.012% | 79.898% |

### global_to_evo

표본 50,000장 · 평가 완료 11-15 20:19:27 (KST)

| 추론 경로 | 해당 경로 단독 Top-1 | τ=1 최종 Top-1 |
| --- | ---: | ---: |
| narrow | 73.534% | 73.534% |
| wide | 81.044% | 79.934% |
| evo_layer_25 | 79.152% | 78.390% |
| evo_layer_50 | 80.630% | 79.576% |
| evo_layer_75 | 81.018% | 79.872% |
| evo_stage_25 | 79.074% | 78.332% |
| evo_stage_50 | 80.580% | 79.534% |
| evo_stage_75 | 81.060% | 79.908% |

## 정확도·연산량 트레이드오프

주 비교는 원본 전체 토큰 3H→6H·τ=1 대비다. GMACs는 이미지당 평균 곱셈-누산 횟수(10억 단위)이며,
여기서는 1 MAC을 곱셈+누산 한 쌍으로 센다(보통 2 FLOPs). 실제 추론 시간과 분리해 보고한다.
원본 대비 정확도 하락을 허용하면서 GMACs를 줄이는지, 같은 GMACs에서 정확도가 높은지를 확인한다.
Global CLS/slow-fast는 같은 유지율에서 기존 Drop과 블록 토큰 예산이 같아 주 연산량은 거의 같다.
목표는 해당 예산에서 정확도 보존이다. GMACs 절감의 대부분은 토큰 수 감소에서 발생한다.

| 모델 / 경로 | 유지율 | Top-1 | 평균 GMACs | 원본 대비 연산량 감소 | 정확도 차이 %p | 깨끗한 batch-1 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official / full | 100% | 80.304% | 2.9438 | 0.00% | +0.000 | 6.215 |
| existing_kd / last_cls | 25% | 78.388% | 1.7010 | 42.22% | -1.916 | 미측정 |
| existing_kd / last_cls | 50% | 79.672% | 2.1014 | 28.61% | -0.632 | 6.565 |
| existing_kd / last_cls | 75% | 79.978% | 2.5179 | 14.47% | -0.326 | 미측정 |
| global_cls / global_cls | 25% | 78.448% | 1.7017 | 42.19% | -1.856 | 미측정 |
| global_cls / global_cls | 50% | 79.678% | 2.1024 | 28.58% | -0.626 | 7.843 |
| global_cls / global_cls | 75% | 79.984% | 2.5192 | 14.42% | -0.320 | 미측정 |
| evo_schedule / evo_layer | 25% | 78.216% | 1.7019 | 42.19% | -2.088 | 미측정 |
| evo_schedule / evo_stage | 25% | 78.194% | 1.7019 | 42.19% | -2.110 | 미측정 |
| evo_schedule / evo_layer | 50% | 79.634% | 2.1025 | 28.58% | -0.670 | 9.183 |
| evo_schedule / evo_stage | 50% | 79.572% | 2.1025 | 28.58% | -0.732 | 8.266 |
| evo_schedule / evo_layer | 75% | 79.894% | 2.5192 | 14.42% | -0.410 | 미측정 |
| evo_schedule / evo_stage | 75% | 79.898% | 2.5192 | 14.42% | -0.406 | 미측정 |
| global_to_evo / evo_layer | 25% | 78.390% | 1.7018 | 42.19% | -1.914 | 미측정 |
| global_to_evo / evo_stage | 25% | 78.332% | 1.7018 | 42.19% | -1.972 | 미측정 |
| global_to_evo / evo_layer | 50% | 79.576% | 2.1023 | 28.59% | -0.728 | 8.964 |
| global_to_evo / evo_stage | 50% | 79.534% | 2.1023 | 28.59% | -0.770 | 8.678 |
| global_to_evo / evo_layer | 75% | 79.872% | 2.5187 | 14.44% | -0.432 | 미측정 |
| global_to_evo / evo_stage | 75% | 79.908% | 2.5187 | 14.44% | -0.396 | 미측정 |

이 수치는 conv·선형층·attention 행렬곱·재활용 projection·선택용 CLS QK·대표 가중합의 분석적 계산이다.
norm/softmax/GELU·elementwise 갱신·정렬·gather/scatter·메모리 이동·CPU 호출 비용은 GMACs에서 제외한다.
따라서 연산량이 줄어도 실제 GPU 시간이 늘 수 있다. GPU 대기·메모리 병목을 분리하려면 profiler가 필요하며,
현재 자료만으로 병목의 원인을 확정하지 않는다. 지연 평균/p95, batch throughput, peak VRAM도 별도 지표다.
기존 저장소 수치(원본 약 2.915 GMACs)는 3H patch embedding을 192채널로 가정했다.
실제 구현은 384채널 embedding을 만든 뒤 첫 블록에서 자르므로 여기서는 모든 경로에 이를 보정했다.
GMACs 수치가 약 0.029 증가한 것은 계산 기준 보정이며 모델 변경 때문이 아니다.
CPU에서 실제 모듈 입출력 형태를 추적해 3H core 1,282,584,576 MACs와 50% 6H core 2,265,169,920 MACs를 확인했다.
증거: `compute_shape_verification.json`. GPU 속도를 CPU에서 측정한 값이 아니다.
원자료: `compute_tradeoff.json`. 그림의 회색 곡선은 원본 모델의 threshold 탐색으로, 같은 연산량에서의 대조군이다.
threshold 곡선은 같은 validation에서 얻은 사후 탐색이며 별도 calibration으로 선택한 배포 성능은 아니다.

![정확도 대 평균 GMACs](/home/misys/OnDevice/results/evo-vit-20261115/accuracy_compute_tradeoff.png)

그림은 생성 시점에 평가가 완료된 모델만 포함한다. 위 표는 후속 평가 완료 시 자동 추가된다.

## GPU 추론 시간 재측정

### latency-clean-now.json

상태: **complete**

같은 세션·같은 seed-42 validation 1,000장·batch 1·τ=1·유지율 50%·FP32/TF32·CPU thread 1.
50장 warmup, 방법 순서를 섞어 3회 반복, 각 반복 평균의 중앙값과 범위를 표시한다.
전처리/H2D/최종 argmax 제외, 분기·선택 부가 연산 포함. full 대조군은 선택용 attention을 추출하지 않는다.
1초 간격으로 다른 GPU compute 프로세스와 정지 상태, 온도·사용률·전력·클럭을 기록한다.
진행 중인 모델은 최종 가중치가 없으므로 재측정 표에 포함하지 않는다.

GPU 공유 작업이 관측되지 않은 측정이다. 메모리에 남은 학습 프로세스는 SIGSTOP 상태로 검증했다.

| 가중치 / 경로 | 평균의 중앙값 ms | 반복 평균 범위 ms |
| --- | ---: | --- |
| official/full | 6.663 | 6.229–6.982 |
| existing_kd/full | 6.931 | 6.380–7.014 |
| existing_kd/last_cls | 7.022 | 6.881–7.050 |
| existing_kd/global_cls | 8.143 | 7.779–8.178 |
| existing_kd/evo_layer | 9.611 | 9.511–9.770 |
| existing_kd/evo_stage | 9.052 | 8.650–9.675 |
| global_cls/full | 6.608 | 6.603–6.952 |
| global_cls/last_cls | 6.810 | 6.713–7.424 |
| global_cls/global_cls | 8.415 | 8.160–8.429 |

기존 학습 재개: 11-15 13:46:24 (KST).

### latency-clean-final.json

상태: **complete**

같은 세션·같은 seed-42 validation 1,000장·batch 1·τ=1·유지율 50%·FP32/TF32·CPU thread 1.
50장 warmup, 방법 순서를 섞어 3회 반복, 각 반복 평균의 중앙값과 범위를 표시한다.
전처리/H2D/최종 argmax 제외, 분기·선택 부가 연산 포함. full 대조군은 선택용 attention을 추출하지 않는다.
1초 간격으로 다른 GPU compute 프로세스와 정지 상태, 온도·사용률·전력·클럭을 기록한다.
진행 중인 모델은 최종 가중치가 없으므로 재측정 표에 포함하지 않는다.

GPU 공유 작업이 관측되지 않은 측정이다. 메모리에 남은 학습 프로세스는 SIGSTOP 상태로 검증했다.

| 가중치 / 경로 | 평균의 중앙값 ms | 반복 평균 범위 ms |
| --- | ---: | --- |
| official/full | 6.215 | 6.003–6.641 |
| existing_kd/full | 6.251 | 6.120–6.608 |
| existing_kd/last_cls | 6.565 | 6.122–7.005 |
| existing_kd/global_cls | 7.266 | 7.055–7.498 |
| existing_kd/evo_layer | 8.617 | 8.446–9.221 |
| existing_kd/evo_stage | 8.774 | 8.515–8.939 |
| global_cls/full | 6.425 | 5.898–6.519 |
| global_cls/last_cls | 6.661 | 6.367–6.957 |
| global_cls/global_cls | 7.843 | 7.146–8.212 |
| evo_schedule/full | 6.267 | 6.070–6.598 |
| evo_schedule/evo_layer | 9.183 | 8.709–9.433 |
| evo_schedule/evo_stage | 8.266 | 8.206–8.742 |
| global_to_evo/full | 6.133 | 6.079–6.566 |
| global_to_evo/evo_layer | 8.964 | 8.698–9.088 |
| global_to_evo/evo_stage | 8.678 | 8.399–8.981 |

## 과거 GPU 추론 시간 기록

오염된 과거 측정은 보존하며, 위 재측정이 완료되면 위 수치를 우선 사용한다.
동일 모델·동일 seed 순서의 1,000장, warmup 50장, batch 1, τ=1, 유지율 50%.
방법 순서를 순환하고 CUDA 동기화 wall-clock으로 측정한다. 전처리·H2D 제외, 분기 동기화 포함.
사전 비교(existing_kd)는 다른 분류기 학습과 GPU를 공유했고 full 대조군도 마지막 CLS attention을 추출했다.
따라서 사전 비교와 이후 평가의 절대 시간을 직접 비교하지 않는다. 재학습 가중치 평가는 full 대조군의
선택용 attention 추출을 생략한다. 동시 프로세스가 관측되면 전용 GPU에서의 속도로 해석하지 않는다.

| 가중치 | 경로 | 평균 ms | 측정 전후 다른 GPU 프로세스 |
| --- | --- | ---: | --- |
| existing_kd | full | 46.632 | 있음(별도 관측 기록) |
| existing_kd | last_cls | 47.313 | 있음(별도 관측 기록) |
| existing_kd | global_cls | 49.540 | 있음(별도 관측 기록) |
| existing_kd | evo_layer | 57.024 | 있음(별도 관측 기록) |
| existing_kd | evo_stage | 53.492 | 있음(별도 관측 기록) |
| global_cls | full | 7.237 | 없음(전후 확인) |
| global_cls | last_cls | 7.523 | 없음(전후 확인) |
| global_cls | global_cls | 8.765 | 없음(전후 확인) |
| evo_schedule | full | 7.043 | 없음(전후 확인) |
| evo_schedule | evo_layer | 10.336 | 없음(전후 확인) |
| evo_schedule | evo_stage | 9.720 | 없음(전후 확인) |
| global_to_evo | full | 7.280 | 없음(전후 확인) |
| global_to_evo | evo_layer | 10.688 | 없음(전후 확인) |
| global_to_evo | evo_stage | 10.058 | 없음(전후 확인) |

## 실행 시각·소요 시간

시각은 KST, 시간은 프로세스 준비·학습/평가·저장을 포함한다.

| 단계 | 시작 KST | 종료 KST | 경과 | 상태 |
| --- | --- | --- | --- | --- |
| existing_kd: validation | 11-15 06:06:23 | 11-15 06:21:23 | 00:15:00 | complete |
| global_cls: KD train | 11-15 06:21:23 | 11-15 10:44:24 | 04:23:00 | complete |
| global_cls: validation | 11-15 10:44:24 | 11-15 10:49:54 | 00:05:30 | complete |
| evo_schedule: KD train | 11-15 10:49:54 | 11-15 15:30:24 | 04:40:30 | complete |
| evo_schedule: validation | 11-15 15:30:24 | 11-15 15:36:24 | 00:06:00 | complete |
| global_to_evo: KD train | 11-15 15:36:26 | 11-15 20:14:01 | 04:37:34 | complete |
| global_to_evo: validation | 11-15 20:14:01 | 11-15 20:20:01 | 00:06:00 | complete |

## 검증·해석·재실행

기존 단위 테스트 40개와 실제 GPU에서 기존 Drop 출력 일치·전체 토큰 경로 일치·BF16 역전파 검증을 통과했다.
추가한 학생 초기화는 공식 Teacher를 복사한 뒤 수행하므로 Teacher 가중치는 바꾸지 않는다.
이어받기 경로는 실제 train 이미지로 2 update·역전파·체크포인트 저장을 확인했다.
검증 시 초기 가중치 SHA-256은 Global CLS 평가 summary의 가중치 SHA-256과 일치했다.
각 결과는 같은 validation을 재사용한 단일 seed 실험이다. 작은 정확도 차이를 확정적 개선으로 보지 않는다.
추가 실험의 개선은 학습 초기화와 누적 6에폭의 영향을 함께 포함한다. 다른 seed·동일 학습량 대조군은 수행하지 않았다.

```bash
.venv/bin/python baseline/run_evolution_warm_start.py
```

위 명령은 기존 runner의 잠금이 해제되고 기존 slow-fast 평가가 완료된 뒤 추가 실험을 실행한다.
완료 단계는 재실행하지 않으며 중단 학습은 optimizer·schedule·RNG·batch 위치를 복원한다.
worker augmentation의 bit 단위 재현은 보장하지 않는다. Git commit·push 없음.

원자료: `/home/misys/OnDevice/results/evo-vit-20261115`. 기존 실행 명령·종료 코드: `study.json`, 추가 실행: `warm_start_study.json`.
각 실험의 `train/arguments.json`, `train/train_log.json`, `train/complete.json`, `eval/summary.json`,
`eval/per_image.pt`에 설정·진행·완료·이미지별 결과를 보존한다.
`global_to_evo/train/arguments.json`에는 초기 가중치 경로와 SHA-256을 기록한다.

[Evo-ViT 논문](https://arxiv.org/abs/2108.01390) ·
[공식 코드](https://github.com/YifanXu74/Evo-ViT/blob/main/deit/evo_deit.py)
