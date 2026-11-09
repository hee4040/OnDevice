# 4번 과제: 3H를 보존하는 6H 부분 파인튜닝

## 과제 범위

질문은 **같은 토큰을 같은 수만큼 계산하면서, 일부 파라미터만 학습해 Drop 6H의 정확도를 회복할 수 있는가?** 다. GPU 실행 방식·블록별 토큰 수·이미지별 유지율은 변경하지 않는다.

| 담당 실험 | 바꾸는 대상 | 이번 4번에서 고정할 것 |
| --- | --- | --- |
| 1. GPU 병목 분석 | 연산별 시간 조사 | 결과·측정 도구를 참고하고 재수행하지 않음 |
| 2. GPU 최적화 Drop | compile·CUDA Graphs·정밀도 | 담당자의 도구를 재사용; 정확도 비교는 우선 같은 FP32로 통일 |
| 3. Layer-wise Drop | 블록별 토큰 유지율 | 모든 6H 블록에서 같은 토큰 집합·같은 유지율 |
| **4. Partial Fine-tuning** | **학습할 파라미터와 stage별 적용 범위** | **여기에서만 변경** |
| 5. 이미지별 유지율 | 이미지에 따라 25/50/75% 선택 | 주 평가는 모든 routed 이미지에서 50% 고정 |

3H CLS attention의 Top-K 선택, 12개 블록 실행, entropy τ=1을 유지한다. 임계값 calibration이나 학습형 토큰 선택·병합을 이번 주제로 추가하지 않는다. 일부 블록의 **가중치를 학습**하는 비교는 4번이며, 블록마다 **토큰 수를 바꾸는** 3번과 구분된다.

## 새 브랜치에서 확인·가져온 것

검토 당시 원격 commit:

- `experiment/h1-diagnosis`: `9d8f655` — T4 GPU 최적화 실험과 단계 합산/실제 pipeline 비교.
- `codex/layer-wise-drop`: `8bd6006` — Layer-wise Drop 코드·Colab 절차. 해당 문서에는 아직 새 성능 결과가 없다고 명시돼 있다.

현재 `experiment/drop-ft`에 필요한 파일만 가져왔다. 전체 브랜치를 merge하거나 기존 학습·시간 기록 코드를 덮어쓰지 않았다.

| 가져온 파일/기능 | 용도 |
| --- | --- |
| `baseline/optimize_latency.py` | 고정 길이 Top-K 인덱스 gather, 기존 mask 경로와 logits 비교, 담당자의 GPU 측정 도구 |
| `baseline/analyze_exp2.py` | 담당자의 단계 시간·동일 시간 예산 분석 재현 도구 |
| `baseline/gpu_optimization.md`, `baseline/reports/2026-11-08/` | T4 실측·FP16 검증·이미지·pipeline 결과를 로컬 참고 자료로 보존 |
| `baseline/evaluate_drop.py`의 `--fp16` | 기존 output-dir·provenance 기능과 결합해 선택적으로 FP16 평가 가능. 기본 FP32 동작은 유지 |
| `baseline/layerwise_drop.md` | 3번의 고정된 담당 범위·유지율 스케줄 확인용 문서 |

T4의 compile+CUDA Graphs 결과에는 **단계 합산 추정 시간**이 포함돼 있다. 최적화된 실제 end-to-end 시간이 모두 검증됐다고 해석하지 않는다. T4·PyTorch 2.11의 절대 수치를 현재 3070 Ti·PyTorch 2.5.1에 그대로 대입하지 않는다. 자세한 내용은 가져온 GPU 문서의 3-5와 한계를 따른다.

## freeze가 세 가지 다른 의미로 쓰였다

| 구분 | 실제로 고정하는 것 | 다음 블록에서 토큰은 갱신되는가? |
| --- | --- | --- |
| 실패했던 H1 token freeze | 선택하지 않은 토큰의 활성값 | 비선택 토큰은 초기 상태에 남아 계속 K/V에 참여 |
| 토큰 요약·병합 | 여러 토큰을 압축해 개수를 줄임 | 남긴 요약 토큰도 블록마다 갱신 가능 |
| Partial FT의 backbone freeze | optimizer가 바꾸는 모델 가중치 | 모든 남은 토큰은 매 forward에서 정상 갱신 |

따라서 H1의 큰 정확도 붕괴가 곧 backbone을 동결하는 부분 학습도 실패한다는 뜻은 아니다. 요약 역시 stale 토큰을 남기는 freeze와 다르지만 손실적 압축이라 효과가 보장되지는 않는다. 이번 4번에서는 토큰 구조를 유지하고 부분 학습에 집중한다.

## 이전에 아직 하지 않은 실험

기존 KD·CE·Full 보존·Feature 실험은 모두 `AdamW(student.parameters())`로 학생 전체를 학습했다. 동결한 것은 Teacher다. 즉 네 실험은 **손실 함수 비교**였으며 LayerNorm-only, projection-only, 6H 전용 LoRA/Adapter 비교는 아직 하지 않았다.

현재 학생은 총 **22,124,777개** 파라미터다. 3H와 6H는 별도 모델이 아니라 같은 블록·분류기의 파라미터를 공유하므로, 전체 파인튜닝은 3H 출력·분기·토큰 순위도 바꿀 수 있다.

확인된 결과:

- 로컬 전체 KD의 Drop 50% Top-1: 79.672%.
- 전체 6H Top-1은 원본 81.440% → 전체 KD 81.092%로 내려갔다.
- Full 보존 손실은 전체 6H를 81.360%로 회복했으나 Drop 50%는 79.656%였다.

이 결과는 원본 경로 보존과 드롭 적응을 구분해서 볼 필요가 있음을 보여준다. **실제 gradient 충돌을 측정한 결과는 아니므로 원인을 확정하지는 않는다.**

## 추천하는 주 실험

먼저 아래 세 후보를 같은 초기 가중치·학습 데이터·학습량으로 비교한다. 첫 단계에서는 기존 학습의 25/50% batch별 선택도 유지하고, 배포 평가에서 50%를 고정해 **학습 대상만** 바꾼다. 50% 전용 학습은 이후 별도 보조 실험으로 구분한다.

| 실험 | 학습할 부분 | 예상 학습 파라미터 | 질문 |
| --- | --- | ---: | --- |
| 대조군 | 기존 전체 KD | 22,124,777 | 기존 79.672% 재현/동일 조건 기준 |
| P | 재활용 `proj_layer` + `alpha`만 | **74,113** | 3H 결과를 6H에 넣는 변환만 조정해도 회복되는가? |
| P+LN | P + 6H Drop 전용 LayerNorm affine | **93,313** | 줄어든 토큰 집합에 정규화 scale·bias를 적응시키면 더 좋아지는가? |
| P+LN+LoRA | P+LN + 6H Drop 전용 Q/V LoRA, rank 8, 12블록 | **240,769** | 3H를 건드리지 않고 attention 변환에 적응 용량을 주면 개선되는가? |

P의 74,113개와 LayerNorm 19,200개는 실제 로딩한 모델에서 집계했다. LayerNorm은 블록당 norm1/norm2 24개와 최종 norm 1개, 총 25개 × (384 scale + 384 bias)다. LoRA는 독립적인 Q/V에 대해 `12 × 2 × 8 × (384+384) = 147,456`개를 가정한 값이며, 구현 시 실제 집계를 다시 확인한다.

전체 모델 기준 학습 비율은 P 약 0.34%, P+LN 약 0.42%, P+LN+LoRA 약 1.09%다. **학습 파라미터 비율과 추론 GMACs 감소는 다른 값**이다. 기본 Transformer를 그대로 실행하므로 파라미터를 적게 학습한다고 추론 연산이 크게 줄어드는 것은 아니다.

가설은 토큰을 새로 복원하는 것이 아니라, 남아 있는 3H 재활용 표현과 6H 변환을 드롭 조건에 맞게 적응시키면서 공유된 3H 경로의 변화를 막는 것이다. 새로운 정확도 개선치는 아직 없다.

### stage별 적용에서 중요한 구현 조건

- patch embedding·CLS·position embedding·원본 Transformer·분류 head는 동결한다. P는 6H 입력 재활용에만 사용되므로 3H에는 영향을 주지 않는다.
- 원래 공유 LayerNorm affine을 그대로 학습하면 3H도 변한다. **6H Drop에서만 사용하는 별도 affine**을 두고 3H는 원래 값을 사용해야 한다.
- 첫 3H 블록에 들어오는 원본 토큰은 아직 384차원일 수 있다. 단순히 입력 tensor의 마지막 크기로 stage를 판단하지 말고, `keep_shape` 또는 명시적 stage/Drop flag로 구분한다.
- LoRA 역시 6H Drop 경로에서만 적용한다. 공유 QKV에 전역으로 붙이거나 merge해 3H까지 바꾸지 않는다. 적용을 stage별로 유지하거나, 배포 시 별도 fused weight를 쓸 경우 메모리 비용을 보고한다.
- 별도 LayerNorm은 19,200개를 추가 저장한다. P는 기존 파라미터를 학습한다. LoRA도 추가 저장량과 trainable 수를 따로 기록한다.
- 모델을 `eval()`로 두는 것과 gradient를 끄는 것은 다르다. 현재 dropout 없는 forward 조건을 유지하면서 선택한 파라미터의 gradient는 켠다.
- 실제 optimizer step 전후 **3H logits·entropy routing·Top-K 인덱스가 유지되는지** 검증한다. 단순히 3H 평균 정확도만 같다고 보존됐다고 판단하지 않는다.
- P를 학습하면 전체 토큰 6H 경로도 바뀔 수 있으므로 전체 6H 정확도는 계속 측정한다. 3H 보존과 전체 6H 보존을 동일한 것으로 취급하지 않는다.

LayerNorm만 학습하는 비교는 ViT PEFT 문헌에서도 의미 있는 대조군이다. 다만 해당 문헌은 few-shot adaptation이며 이번 ImageNet 드롭 복구의 성공 근거로 직접 해석하지 않는다. [LN-Tune 연구](https://arxiv.org/abs/2304.01917)

LoRA는 저랭크 가중치 업데이트라는 방법을 참고한다. [LoRA 원 논문](https://arxiv.org/abs/2106.09685)

### 선택적 추가 비교: Adapter

LoRA와 비교할 필요가 있으면 6H에만 parallel bottleneck adapter를 추가한다. bottleneck 16, down/up bias 포함, 별도 norm 없이 12블록이면 약 152,256개다. Q/V rank-8 LoRA의 147,456개와 비슷한 학습 파라미터 예산으로 비교할 수 있다. adapter 출력은 0으로 초기화해 학습 전 원본 경로와 일치하는지 확인한다.

Adapter는 추가 forward 연산이 있으므로 시간·추론 GMACs가 완전히 같다고 가정하지 않는다. ViT용 adapter의 구조적 참고는 [AdaptFormer](https://arxiv.org/abs/2205.13535)이며, 해당 논문의 downstream 성능을 이번 문제의 개선치로 예측하지 않는다.

## 꼭 Transformer 파인튜닝이 아니어도 되는 예비 주제

**고정된 3H와 Drop 6H의 출력 보정 head**도 다른 담당자와 겹치지 않는 작은 실험이다.

모델 두 경로와 routing·토큰 수를 모두 고정한 채, routed 이미지의 최종 답만 `z6 + a*z3` 같은 작은 logits 결합 또는 작은 보정 head로 학습한다. 이미 구한 3H 답을 활용해 6H가 원래 맞던 답을 틀리는 경우를 줄일 수 있는지 묻는다. 3H 특징을 다시 Transformer에 넣거나 토큰을 추가하지 않는다.

train의 별도 calibration 자료에서 결합 계수/head를 학습하고 검증에서 비교해야 한다. 검증 정답으로 가중치를 고르면 성능 개선으로 보고하지 않는다. 서로 상관된 모델이라 개선이 없을 수도 있지만, backbone 전체를 반복 학습하기 전에 작은 비용으로 확인할 수 있다. 이는 **최종 답 보정**이며 이미지별 25/50/75% 토큰 수를 결정하는 5번과 구분된다.

## 비교·실행 계획

1. 앞선 전체 KD를 대조군으로 두고 먼저 P와 P+LN의 실행·gradient·3H 보존 검증을 한다.
2. train에서 사전에 고정한 작은 층화 subset과 별도 train calibration으로 학습률을 점검한다. 서로 다른 방법에 전체 FT의 학습률 5e-5만 강제해 불리하게 만들지 않는다. 탐색 후보 수·데이터 예산은 같게 하고 기준을 미리 적는다.
3. 정한 설정으로 전체 train 3에폭을 실행하고, 같은 validation 순서로 3H·전체 6H·Drop 50%를 평가한다. CE·Full 보존·Feature 손실을 동시에 바꾸지 않고 기존 KD 목적을 우선 유지한다.
4. P+LN이 유망하거나 용량 부족이 확인되면 LoRA를 비교하고, 필요할 때만 Adapter를 추가한다. 모든 후보를 무조건 장시간 실행하지 않는다.
5. Top-1·routed Top-1·원본 교정 유지율·3H 예측/분기/Top-K 변화·학습 파라미터 수·추가 저장량·학습 시간·peak memory를 기록한다. 작은 정확도 차이는 paired 비교와 추가 seed로 확인한다.
6. 채택한 가중치만 2번 담당자의 실행 도구로 최종 GPU 평가한다. GPU 최적화 자체를 별도 기여로 다시 주장하지 않는다.

2026-11-09 사용자 요청에 따라 P·P+LN·P+LN+LoRA를 모두 독립 실행하기 시작했다. 학습률 pilot 뒤 각 3에폭을 수행하며, 실제 진행·정확도·시간은 [partial_ft_results.md](partial_ft_results.md)에 자동 기록한다. 실행기는 `baseline/run_partial_ft_study.py`, 원자료는 `results/partial-ft-20261109/`다. Adapter는 이번 세 실험에 포함하지 않는다. Commit·push는 수행하지 않았다.
