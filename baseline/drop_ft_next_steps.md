# Drop fine-tuning 후속 구조 분석 — 2026-11-08

## 판단

현재 결과로 정확도 회복이 불가능하다고 결론낼 수 없다. 고정 τ=1에서 손실을 추가하는 실험은 끝났지만, 토큰 선택·압축 시점·이미지 분기 기준은 학습하지 않았다. 추가 분석에서 **기존 Full 보존 가중치의 임계값 조정만으로 원본 수준의 정확도**를 얻었다. 다만 **GPU batch 1 시간 감소는 아직 달성하지 못했다.**

이번에는 모델·학습 코드를 바꾸거나 새 장시간 학습을 시작하지 않았다. 저장된 이미지별 결과를 재분석하고, 이미 학습한 모델의 GPU 추론을 추가 측정했다. 기존 실험 기록은 [drop_ft_experiments.md](drop_ft_experiments.md)에 보존한다.

## 실제 실행 구조

1. 입력 224×224 → 패치 196개와 CLS 1개. 원래 embedding은 384차원이다.
2. 3H는 각 블록에서 192차원·attention head 3개·MLP hidden 768을 사용한다. **12개 블록 전부**를 지난다.
3. 마지막 3H CLS로 1,000개 클래스 logits를 계산한다. 상위 10개 확률을 재정규화한 entropy로 조기 종료 여부를 결정한다.
4. 6H 입력은 `원본 patch/position embedding + alpha × proj_layer(3H의 마지막 토큰 특징)`이다. 별도 다음 12개 블록을 새로 두는 구조가 아니라, **같은 12개 블록을 384차원·6 heads·MLP hidden 1536으로 다시 사용**한다.
5. 현재 Drop 50%는 3H 마지막 블록의 head 평균 CLS attention으로 패치 98개를 선택한다. CLS와 함께 99개 토큰만 **첫 6H 블록 전에** 남긴다. 같은 토큰 집합이 6H 12개 블록을 모두 지난다.

근거: `third_party/ThinkingViT/timm/models/vision_transformer.py`의 `Mlp.forward`, `Linear.forward`, `QKVLinear.forward`, `Block.forward`, `forward_features`, `entropy`, `forward_head`; `baseline/evaluate_h1_diagnosis.py:65`와 `:73`; `baseline/selective_refinement.py:12`와 `:55`.

드롭은 실제 gather로 구현돼 있고, 위치 embedding은 gather 전에 적용한다. 이전 freeze 방식처럼 갱신되지 않은 토큰을 K/V에 남기는 구조는 아니다. 현재 특징 증류도 남은 토큰의 원래 위치를 Teacher에 대응시키고 있다.

## 기존 실험의 해석을 보완할 점

### 1. 선택기는 직접 학습하지 않았다

`selective_refinement.py:31`에서 QKV를 `detach()`한 뒤 attention 순위를 만들고, Top-K 인덱스로 bool mask를 만든다. 토큰 값과 공유 모델 가중치에는 gradient가 흐르지만, **어떤 토큰을 골라야 하는지에 대한 직접 gradient는 선택 결정으로 흐르지 않는다.** `detach()`만 제거해도 hard Top-K 선택이 미분 가능해지는 것은 아니다.

따라서 이번 CE·Full·Feature 실험은 선택 정책의 유용성을 직접 학습하는 실험이 아니다. 3H 분류에 중요한 패치 순위와 6H에서 추가로 교정하는 데 중요한 패치 순위가 다를 수 있다는 것이 후속 가설이다.

### 2. CE는 3H와 분기 비율도 바꿨다

| 모델 | 3H Top-1 | τ=1에서 6H 진입 비율 | Drop 50% Top-1 |
| --- | ---: | ---: | ---: |
| KD | 73.498% | 36.148% | 79.672% |
| CE | 74.498% | 32.638% | 79.464% |

CE는 3H 정확도를 1.000%p 높였지만, 같은 τ에서 6H로 보내는 이미지도 줄었다. 고정 τ에서의 최종 정확도 차이는 분류 변화와 분기 변화가 함께 반영된 값이다.

저장된 entropy에서 비슷한 추정 계산량으로 맞춘 탐색 결과는 다음과 같다.

| 모델·임계값 | 전체 Top-1 | 6H 비율 | 추정 GMACs |
| --- | ---: | ---: | ---: |
| KD, τ=1.00 | 79.672% | 36.148% | 2.0725 |
| CE, τ=0.91 | 79.762% | 36.180% | 2.0732 |

CE의 차이는 +0.090%p이고, 고정된 두 모델에 대한 이미지별 paired bootstrap 95% 구간은 −0.088~+0.266%p다. **확정적인 개선은 아니지만, CE를 전면 배제할 근거도 약해졌다.** 이 비교는 전체 entropy로 비슷한 계산량을 고른 사후 탐색이다.

### 3. Full 보존은 실제로 전체 경로를 회복했다

KD → Full 보존의 전체 6H 정확도는 81.092 → 81.360%다. 차이 +0.268%p의 이미지별 paired bootstrap 구간은 +0.142~+0.398%p다. 반면 Drop 50%의 −0.016%p 차이는 구간 −0.150~+0.116%p여서 이번 단일 실행만으로 악화라고 단정하기 어렵다.

이 구간은 학습을 여러 seed로 반복한 변동성이 아니라, **이미 학습된 모델들을 고정한 상태의 이미지 표본 변동성**이다.

### 4. 공유 경로 적응과 드롭 손실이 모두 남아 있다

동일 KD 가중치에서 τ=1 전체 토큰 경로는 80.040%, Drop 50%는 79.672%다. 드롭으로 생긴 순손실은 0.368%p다. 해당 routed 이미지에서 전체 토큰만 맞힌 것은 739장, 드롭만 맞힌 것은 555장이다.

원본 τ=1의 80.304%와 KD 전체 토큰 경로 80.040%의 차이도 0.264%p 남는다. 따라서 드롭 정보 손실만 해결하는 것과, 공유 가중치·분기의 원본 성능을 보존하는 것을 함께 봐야 한다.

## 추가 학습 없이 확인한 정확도 회복

각 모델별로 seed-42 validation 순서의 앞 25,000장에서 원본 τ=1 정확도에 도달하는 최소 추정 계산량의 임계값을 0.01 간격으로 탐색했다. Full 보존 모델에서는 **τ=0.79**가 선택됐다.

| 경로 | 앞 25,000장 | 뒤 25,000장 | 전체 50,000장 | 전체 6H 비율 | 추정 GMACs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 원본, 전체 토큰, τ=1 | 80.640% | 79.968% | 80.304% | 36.008% | 2.9149 |
| Full 보존, Drop 50%, τ=1 | 79.856% | 79.456% | 79.656% | 36.124% | 2.0720 |
| Full 보존, Drop 50%, τ=0.79 | 80.640% | 80.060% | **80.350%** | 45.620% | **2.2871** |

현재 가중치만으로 원본에 가까운 정확도를 얻고, 기존 분석 공식의 MACs는 약 21.5% 적다. 그러나 전체 차이 +0.046%p의 paired bootstrap 구간은 −0.142~+0.232%p다. 원본보다 우수하다고 주장하지 않는다.

이것은 저장된 모든 이미지의 실제 경로별 정오답과 entropy를 재조합한 결과다. 새 전체 validation forward를 실행한 결과로 표시하지 않는다. 임계값 선택과 여러 모델 비교가 포함된 **탐색 결과**이며, 다음 학습에서는 train의 별도 calibration split과 새로운 확인 데이터로 선택·검증을 분리해야 한다. 기존 뒤 25,000장을 반복적으로 완전한 미사용 테스트 세트로 취급할 수는 없다.

MACs는 `plot_h1_diagnosis.py`의 기존 해석적 공식이다. norm·softmax·선택 부가 비용을 제외한 추정치이며 GPU 시간과 다르다.

## GPU 시간은 별도 문제다

추가 GPU 측정: 동일 세션, RTX 3070 Ti Laptop, FP32·TF32 허용, 같은 validation 1,000장, batch 1, 50장 warmup, 방법 순서를 바꿔 3회 반복했다. CUDA 동기화 wall-clock으로 gate를 포함하고 전처리·H2D·최종 argmax는 제외했다.

| 경로 | 반복별 평균의 중앙값 | 반복 범위 | 1,000장 내 6H 비율 |
| --- | ---: | ---: | ---: |
| 원본, τ=1 | 10.662ms | 10.620~11.127ms | 33.2% |
| Full 보존, Drop 50%, τ=1 | 11.978ms | 10.756~12.273ms | 33.6% |
| Full 보존, Drop 50%, τ=0.79 | 12.277ms | 12.222~12.394ms | 41.9% |

**임계값 보정은 이번 구현의 GPU batch-1 속도 해결책이 아니다.** 더 자주 6H를 실행하기 때문이다. 이전 날짜 기록과 이번 절대 시간도 다르므로, 서로 다른 세션의 시간 차이를 학습 효과로 해석하면 안 된다.

추가 측정 전체 실행 시간은 129.22초(입력 준비 포함)다. 정확한 시작·종료 시각과 원자료는 `results/drop-ft-20261106/threshold_latency_probe.json`, 실행 코드는 같은 폴더의 `route_probe.py`에 보존했다.

## 후속 방법의 우선순위

### A. GPU 속도: 실행 요청 비용을 먼저 줄인다

현재 선택 단계는 hook 등록·별도 CLS attention 계산·Top-K → bool mask → `nonzero()` → gather를 사용한다. gate는 `.item()`으로 GPU 결과를 기다린다. QKV 계층도 매 forward에서 Python 인덱스 목록과 가중치 indexing을 사용한다. 토큰 수가 줄어도 12개 블록의 연산 호출 수는 유지된다.

프로파일링으로 실제 병목을 구분하고, 다음을 **원본과 드롭 경로에 모두** 적용한 뒤 비교한다.

1. 고정 K의 Top-K 인덱스를 직접 gather에 사용한다. 현재처럼 원래 위치 순서가 필요하면 고정 길이 인덱스를 정렬한다.
2. 선택용 score를 forward의 명시적 출력으로 제공한다. 매번 hook을 등록하는 경로를 정리한다.
3. 3H·6H를 고정 width로 실행하도록 특수화한다. `.item()` 분기 바깥의 각 단계를 별도로 compile/CUDA Graph 적용 후보로 삼는다.
4. 같은 세션·같은 정밀도에서 원본과 비교한다. batch 1과 batch 16/32의 throughput은 분리해서 기록한다.

CUDA Graph는 작은 GPU 연산들의 CPU 실행 요청 비용을 줄이는 방법이다. 정적 shape·제어 흐름이 필요한 제약을 고려해야 한다. [PyTorch 공식 설명](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

추론 엔진 개선 뒤에도 batch-1 시간이 줄지 않으면, 6H 블록 수를 줄이는 구조 증류나 6H 진입 자체를 줄이는 gate가 더 직접적인 후보다. 이때도 정확도 보존은 새로 검증해야 한다.

### B. 정확도·계산량: 6H의 학습형 토큰 선택

현재 고정 3H CLS 순위 대신 작은 predictor가 토큰의 보존 가치를 학습하게 한다. 초기 6H 블록 뒤 또는 소수의 고정 지점에서 선택하고, 학습 때 미분 가능한 정책/attention masking을 사용하며 추론 때 실제 gather로 토큰을 줄인다. 처음에는 지점을 하나로 제한해 선택 부가 비용을 측정한다.

**가장 직접적인 논문 근거**는 ThinkingViT 논문 부록 G다. 저자들은 ThinkingViT 두 번째 단계에 DynamicViT를 결합한 결과를 보고한다. 이것이 현재 체크포인트·3에폭·GPU batch 1에서도 같은 개선을 보장하는 것은 아니다. 로컬 checkout에는 해당 학습형 predictor가 구현돼 있지 않다. [ThinkingViT 논문·부록 G](https://arxiv.org/html/2507.10800v3)

DynamicViT 공식 구현은 별도 score predictor와 학습 시 Gumbel-Softmax 정책, 추론 시 인덱스 선택을 사용한다. 현재 attention의 `detach()`만 지우는 수정과는 다르다. [공식 모델 코드](https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py)

### C. 정보 보존: 버릴 토큰을 요약·병합한다

학습 가능한 빈 글로벌 토큰을 추가하는 것보다, **실제로 버릴 토큰들의 값을 압축해 전달**하는 것이 현재 문제에 더 직접적이다.

- 작은 실험: 총 99개 토큰 예산을 유지하면서 `CLS 1 + 선택 패치 97 + 버릴 패치의 가중 요약 1`을 사용한다. 드롭과 같은 토큰 수에서 요약·선택 비용도 별도로 측정한다.
- 여러 영역이 섞이는 문제가 있으면 소수의 공간별 요약이나 유사한 토큰끼리의 병합을 비교한다. 압축은 손실적이며 정보 복원이 보장되는 것은 아니다.
- 요약 토큰도 이후 6H 블록에서 함께 갱신한다. 새 토큰에 위치 embedding을 중복 적용하지 않는다.

EViT는 선택하지 않은 토큰을 CLS attention으로 가중합해 후속 계산에 남긴다. [EViT 공식 구현](https://github.com/youweiliang/evit/blob/master/evit.py)

ToMe는 유사 토큰을 점진적으로 병합하고, 크기를 반영한 가중 평균과 선택적 proportional attention을 사용한다. [ToMe 공식 병합 코드](https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py), [attention 구현](https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py)

이 모델에 표준 ToMe patch를 그대로 적용하면 custom `forward(x, p)`와 맞지 않고 3H의 재활용 토큰 수까지 바꿀 수 있다. **3H의 197개 토큰은 보존하고, 6H 재활용 입력을 만든 뒤 6H에서만 압축**하는 구현이 필요하다.

관련된 2026년 MergeOver 연구는 SReT에 병합을 적용했으며, GPU batch 1 throughput은 오히려 감소하고 batch 16에서는 증가했다고 보고한다. ThinkingViT에 대한 검증은 아니지만, 병합 역시 batch-1 가속을 당연시할 수 없다는 비교 근거다. [MergeOver 논문](https://arxiv.org/abs/2608.13141)

### D. 공유 가중치: 3H를 보존하면서 6H를 적응시킨다

3H 출력·gate·토큰 순위를 고정하고 6H에서만 사용하는 파라미터 부분, 재활용 projection, 또는 6H 전용 adapter를 학습하는 대조군이다. 공유 가중치의 목표 간 충돌 여부는 아직 gradient 분석으로 입증하지 않았으므로 이 방법의 효과는 가설이다.

단순히 전체 `blocks`를 freeze하면 6H 적응도 제한된다. 텐서 내부의 3H 사용 영역만 보존하려면 gradient/optimizer 처리를 구분해야 한다. AdamW weight decay나 기존 momentum도 보존 영역을 바꿀 수 있다. 여러 optimizer step 뒤 **3H logits와 선택 인덱스가 실제로 유지되는지** 검사해야 한다.

6H 전용 LoRA를 모든 공유 가중치에 한꺼번에 merge하면 3H도 달라질 수 있다. stage별 적용과 배포 시 가중치 캐시·메모리 비용까지 함께 설계해야 한다. 기존 채널의 6H 전용 영역만 조정하는 방식은 추가 추론 연산 없이 시도할 수 있는 대안이다.

### E. GPU 진입 횟수: 교정 이득을 예측하는 gate

현재 gate는 3H entropy만 사용한다. 고정된 3H와 Drop 6H의 train 예측에서 `3H 오답→6H 정답`과 `3H 정답→6H 오답`을 구분해, 3H CLS·상위 logits·margin으로 **기대 교정 이득**을 예측하는 작은 gate를 학습할 수 있다. 성공하면 6H 12블록의 실행 자체를 줄이므로 GPU batch 1에 직접적인 절약이 된다. gate 자체 비용과 별도 train calibration이 필요하다.

저장된 KD 결과에서 정답을 알고 경로를 고르면 83.118%라는 oracle 상한이 나온다. 이것은 실제 학습형 gate의 성능 예측이 아니며 배포 가능한 결과도 아니다. 다만 entropy 분기에 남은 선택 여지를 진단하는 값이다.

## 권하는 다음 실험 순서

1. **배포 경로 프로파일링·특수화와 임계값 calibration**을 먼저 한다. 현재 모델로 정확도 목표와 GPU 시간 목표를 동시에 달성할 수 있는지 짧게 확인한다.
2. 정확도·MACs 목표에는 **6H 학습형 선택기**를 주요 구조 실험으로 삼고, 총 토큰 수를 맞춘 **요약 토큰 병합**을 단순 대조군으로 둔다.
3. 50% 유지가 주 목표이면 50% 전용 학습과 기존 25/50% 공동 학습을 구분한다. 이후 더 긴 학습이 필요하면 먼저 후보를 좁힌다. EViT 공식 fine-tuning 예시는 30에폭이지만, 이를 현재 하드웨어·학습률에 그대로 복사하지 않는다. [EViT 학습 예시](https://github.com/youweiliang/evit/blob/master/finetune.sh)
4. 새 train calibration split으로 임계값을 정하고, 같은 정확도에서 MACs·batch-1 latency·batch throughput을 함께 비교한다. 한 지표의 개선을 다른 지표의 개선으로 표시하지 않는다.

## 이번 분석 원자료

`results/drop-ft-20261106/` 아래:

- `structural_diagnosis.json`: 경로별 정확도, 분기 변화, paired bootstrap, 드롭으로 얻고 잃은 정답 수.
- `threshold_diagnosis.json`: 각 모델 τ=0~2.30, 간격 0.01의 정확도·routing·추정 MACs.
- `threshold_comparison.json`: 원본과 Full τ=0.79의 이미지별 대응 비교.
- `gate_diagnosis.json`: CE 비용 보정 비교와 gate oracle 진단.
- `threshold_latency_probe.json`, `route_probe.py`: 추가 GPU 추론 시간과 재현 코드.

저장된 평가의 이미지 순서는 모두 seed 42의 같은 validation 순서다. 새 학습·Git commit·GitHub push는 수행하지 않았다.
