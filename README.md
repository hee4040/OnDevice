# OnDevice: 토큰 드롭 후 정확도 회복 실험

ThinkingViT(3H → 6H)에서 토큰을 줄였을 때의 정확도 손실을 **지식 증류와 부분 파인튜닝으로 회복할 수 있는지** 비교한 저장소입니다. 이 문서는 `experiment/drop-ft` 브랜치에서 2026년 11월 6~9일 수행한 작업을 요약합니다.

**주요 결과:** 고정 Drop 50%·τ=1에서 기존 전체 KD가 **79.672%**로 가장 높았습니다. 부분 학습에서는 **projection + LayerNorm + LoRA가 79.512%**로 가장 높았으며, 전체의 약 **1.09% 파라미터만 학습하면서 3H 보존 검사를 통과**했습니다.

## 무엇을 했나

| 순서 | 질문 | 수행한 작업 | 결과 |
| --- | --- | --- | --- |
| 1. 기존 실험 재현 | 다른 GPU에서도 기존 결과가 재현되는가? | 원본 평가와 기존 KD 파인튜닝을 로컬 GPU에서 재실행 | 원본 τ=1은 80.304%, KD Drop 50%는 79.672%로 기존 4090 기록과 유사 |
| 2. 손실 함수 비교 | KD에 다른 학습 신호를 더하면 정확도가 회복되는가? | 정답 CE, 전체 토큰 경로 보존, 중간 특징 증류를 각각 추가 | 고정 τ=1에서 기존 KD의 Drop 50% 정확도를 넘은 방법은 없음 |
| 3. 구조·분기 분석 | 가중치를 다시 학습하지 않고 개선할 여지가 있는가? | 저장된 이미지별 예측으로 임계값 분석, 추가 GPU 시간 측정 | Full 보존 모델의 τ 조정으로 80.350%를 얻었으나 GPU batch-1 가속은 달성하지 못함 |
| 4. 부분 파인튜닝 | 3H를 유지하면서 일부 파라미터만 학습할 수 있는가? | projection, 6H 전용 LayerNorm, 6H 전용 LoRA를 독립 비교 | LoRA 구성이 부분 학습 중 최고. 전체 KD보다 0.160%p 낮지만 학습 파라미터를 크게 줄임 |

이번 담당 범위는 팀 실험의 **4번 Partial Fine-tuning**입니다. GPU 최적화, 블록별 유지율 변경, 이미지별 토큰 유지율 선택은 별도 담당자의 작업입니다.

## 모델과 비교 조건

ThinkingViT는 작은 **3H 경로**로 먼저 분류하고, 불확실한 이미지만 **6H 경로**로 다시 계산합니다. 두 경로는 같은 Transformer 가중치를 공유합니다. 6H 입력에는 3H의 마지막 특징을 projection으로 변환해 더하는 **Token Recycling**을 사용합니다.

이 브랜치의 Drop은 3H 마지막 블록의 CLS attention으로 패치를 고른 뒤, 나머지를 6H 입력에서 실제로 제거합니다. **Drop 25/50/75%는 제거 비율이 아니라 남기는 패치 비율**입니다. Drop 50%는 패치 98개와 CLS 1개, 총 99개 토큰을 남깁니다.

| 항목 | 이번 실험의 공통 조건 |
| --- | --- |
| 모델·입력 | ThinkingViT 3H → 6H, 12블록, 224×224 |
| 초기 가중치 | 공식 일반 EMA 체크포인트. 각 방법을 같은 가중치에서 독립적으로 시작 |
| 학습·평가 데이터 | ImageNet-1K train 1,281,167장 / validation 50,000장 |
| 주 지표 | **Drop 50%·τ=1의 전체 validation Top-1**. Top-1은 최상위 예측이 정답인 비율 |
| 학습 | 3에폭, 유효 batch 128, micro-batch 32의 gradient accumulation, seed 42 |
| 학습 유지율 | batch마다 25% 또는 50%를 선택. 배포 평가는 유지율별로 고정 |
| 최적화·정밀도 | AdamW, weight decay 0, BF16 학습 / FP32·TF32 허용 평가 |
| 로컬 환경 | RTX 3070 Ti Laptop 8GB, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4 |

아래의 **전체 6H**는 모든 이미지를 드롭 없이 6H까지 계산한 정확도입니다. 조기 종료를 포함한 Drop 50%·τ=1과 서로 다른 경로의 지표입니다.

## 1. 기존 KD 재현

4090에서 파인튜닝한 가중치는 로컬에 없었습니다. 대신 **공식 사전학습 EMA를 Teacher와 Student의 시작점으로 사용**해 기존 KD 코드를 새로 실행했습니다. 랜덤 초기화부터 학습한 것은 아닙니다.

Teacher는 고정하고, Student의 3H 출력과 Drop 6H 출력이 Teacher의 3H·전체 6H 출력을 각각 따르도록 학습했습니다. 손실은 `0.5 × KL(3H) + 0.5 × KL(Drop 6H)`, 증류 온도는 1입니다.

| 조건 | 기존 RTX 4090 기록 | 로컬 RTX 3070 Ti 결과 |
| --- | ---: | ---: |
| 원본, 드롭 없이 τ=1 | 80.304% | 80.304% |
| 학습 전 Drop 50%, τ=1 | 78.610% | 78.610% |
| KD 학습 후 Drop 50%, τ=1 | 79.650% | **79.672%** |

원본과 학습 전 Drop 50%의 정확도는 같았고, KD 재학습 결과도 0.022%p 차이로 유사했습니다. 다만 재학습·micro-batch·이전 4090 실행의 중단/재개 조건이 달라, KD의 차이를 GPU만의 영향으로 해석하지 않습니다.

## 2. 손실 함수를 바꾼 전체 파인튜닝

이 단계는 Student **전체 가중치를 학습**하면서 추가 손실만 바꾼 비교입니다. 학습률은 모두 5e-5이며, 네 방법을 각각 공식 EMA에서 시작했습니다.

| 실험 | 기존 KD에 추가한 내용 | Drop 50%·τ=1 | KD 대비 차이 | 전체 6H |
| --- | --- | ---: | ---: | ---: |
| **기존 KD** | 없음 | **79.672%** | 기준 | 81.092% |
| CE 추가 | 3H·Drop 6H에 정답 라벨 CE 추가 | 79.464% | −0.208%p | 80.938% |
| Full 경로 보존 | Student 전체 6H도 Teacher 전체 6H에 맞춤 | 79.656% | −0.016%p | **81.360%** |
| Feature 증류 | 4·8·12번째 블록의 동일 위치 토큰 특징을 맞춤 | 79.626% | −0.046%p | 81.124% |

**고정 τ=1에서는 기존 KD를 넘지 못했습니다.** 다만 Full 경로 보존은 전체 6H 정확도를 회복했습니다. 사전에 정한 조합 채택 기준을 만족한 항목이 없어 추가 조합 학습은 생략했습니다. 상세 조건·이미지별 비교·시간은 [첫 실험 기록](baseline/drop_ft_experiments.md)에 있습니다.

## 3. 3H를 보존하는 부분 파인튜닝

이번 단계에서는 원본 backbone을 동결하고, **6H 적응에 필요한 일부 파라미터만 학습**했습니다. 아래 구성은 누적되지만, 학습은 각 행마다 공식 EMA에서 독립적으로 시작했습니다.

- **P:** 3H 특징을 6H 입력으로 바꾸는 projection과 재활용 계수 `alpha`만 학습.
- **P+LN:** P에 더해 Drop 6H에서만 사용하는 별도 LayerNorm의 scale·bias를 학습.
- **P+LN+LoRA:** P+LN에 더해 12개 블록의 Q/V에 Drop 6H 전용 LoRA를 적용. rank와 alpha는 8.

학습률은 validation 대신 train의 별도 pilot 20,000장과 calibration 2,000장으로 선택했습니다. 후보는 5e-5·2e-4·1e-3이고, 각 200 update 후 calibration의 routed Teacher KL이 가장 낮은 값을 골랐습니다. 이후 다시 초기화해 전체 train을 3에폭 학습했습니다.

| 실험 | 학습 파라미터 | 전체 모델 대비¹ | 선택 lr | Drop 50%·τ=1 | 전체 KD 대비 | 전체 6H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 전체 KD | 22,124,777 | 100% | 5e-5 | **79.672%** | 기준 | 81.092% |
| P | 74,113 | 0.34% | 1e-3 | 79.268% | −0.404%p | 80.580% |
| P+LN | 93,313 | 0.42% | 2e-4 | 79.358% | −0.314%p | 81.178% |
| **P+LN+LoRA** | **240,769** | **1.09%** | 2e-4 | **79.512%** | **−0.160%p** | **81.212%** |

¹ 기준 모델의 22,124,777개 대비 학습할 파라미터 수입니다. 별도 LayerNorm과 LoRA의 추가 저장량이 있으므로 전체 모델 파일이 이 비율로 작아지는 것은 아닙니다.

**부분 학습 중에서는 LoRA 구성이 가장 좋았지만, 정확도 최고값은 전체 KD였습니다.** 학습할 파라미터를 제한하면서 기존 정확도에 얼마나 가까워지는지 확인한 결과입니다. 부분 학습의 lr는 별도 탐색으로 선택했으므로 학습 파라미터 차이만의 순수 인과 비교로 주장하지 않습니다.

3H 보존은 다음과 같이 확인했습니다.

- checkpoint마다 동결 파라미터 hash와 고정 입력의 3H logits·attention·entropy·routing·Top-K 동일성을 검사했습니다.
- 전체 validation 50,000장에서 원본과 entropy 차이 0, 3H 정오답 불일치 0, routing 불일치 0을 확인했습니다.
- 모든 방법에서 12개 블록과 고정 토큰 집합을 유지했습니다. 블록별·이미지별 유지율은 바꾸지 않았습니다.

실패했던 H1의 **토큰 상태 freeze**와 여기서의 **가중치 동결**은 다릅니다. 부분 학습에서도 남아 있는 토큰의 활성값은 매 블록에서 정상 갱신됩니다. 설계는 [부분 학습 계획](baseline/partial_ft_plan.md), 실제 정확도·학습률·보존 검사·메모리는 [부분 학습 결과](baseline/partial_ft_results.md)에 있습니다.

## 4. 추가 분석: 임계값 보정은 정확도와 GPU 시간에 다르게 작용

학습 후 confidence가 달라지므로 τ를 고정한 비교만으로 모든 가능성을 판단할 수는 없습니다. 저장된 이미지별 예측을 재분석한 탐색에서는 다음 결과를 얻었습니다.

| 경로 | Top-1 | 추정 GMACs | 해석 |
| --- | ---: | ---: | --- |
| 원본, 드롭 없이 τ=1 | 80.304% | 2.9149 | 기준 |
| Full 보존 + Drop 50%, τ=0.79 | 80.350% | 2.2871 | 원본 수준 정확도, 추정 MACs 약 21.5% 감소 |
| CE + Drop 50%, τ=0.91 | 79.762% | 2.0732 | 기존 KD(79.672%, 2.0725 GMACs)와 비슷한 비용에서의 탐색 결과 |

임계값 보정은 **저장된 예측의 사후 분석**이며, 새 모델의 독립 검증 성능으로 취급하지 않습니다. GPU 시간은 별도로 측정했습니다. 같은 세션의 batch-1 비교에서 원본 약 10.66ms, Full 보존 + Drop 50%·τ=0.79는 약 12.28ms로 가속되지 않았습니다. **GMACs 감소와 실제 latency 감소는 구분해야 합니다.** 상세 분석과 한계는 [구조·후속 분석](baseline/drop_ft_next_steps.md)을 참고하세요.

## 소요 시간

단위는 시:분:초입니다. 개별 학습 시간에는 모델·로더 준비와 저장이 포함되고, 평가 시간은 별도입니다.

| 단계 | 소요 시간 |
| --- | ---: |
| train 다운로드·추출·개수 검증 | 18:49:41 |
| 전체 KD 재학습 | 03:58:48 |
| CE 추가 학습 | 03:54:07 |
| Full 경로 보존 학습 | 06:32:28 |
| Feature 증류 학습 | 04:00:24 |
| **첫 실험 묶음 전체** — 데이터 준비·위 네 학습·평가·시간 측정 포함 | **37:34:01** |
| 부분 학습 P | 02:58:46 |
| 부분 학습 P+LN | 02:59:46 |
| 부분 학습 P+LN+LoRA | 03:12:51 |
| **부분 학습 묶음 전체** — pilot·calibration·세 본 학습·평가 포함 | **09:40:46** |

validation 평가는 모델당 약 4분이었습니다. 부분 학습 묶음은 이미 준비한 데이터를 사용해 다운로드 시간을 다시 합산하지 않았습니다. 각 단계의 KST 시작·종료 시각은 두 결과 MD에 있습니다.

## 문서·코드 안내

| 목적 | 문서 또는 실행기 |
| --- | --- |
| KD 재현·손실 비교의 전체 결과 | [drop_ft_experiments.md](baseline/drop_ft_experiments.md) |
| 구조 분석·임계값 보정·후속 방법 조사 | [drop_ft_next_steps.md](baseline/drop_ft_next_steps.md) |
| 4번 과제의 범위·부분 학습 설계 | [partial_ft_plan.md](baseline/partial_ft_plan.md) |
| 부분 학습 세 방법의 최종 결과 | [partial_ft_results.md](baseline/partial_ft_results.md) |
| 첫 실험 묶음 실행 | [run_drop_ft_study.py](baseline/run_drop_ft_study.py) |
| 부분 학습 묶음 실행 | [run_partial_ft_study.py](baseline/run_partial_ft_study.py) |
| 공통 학습·평가 / 부분 학습 모듈 | [finetune_drop.py](baseline/finetune_drop.py), [evaluate_drop.py](baseline/evaluate_drop.py), [partial_tuning.py](baseline/partial_tuning.py) |
| 팀원의 GPU 최적화 결과·도구 | [gpu_optimization.md](baseline/gpu_optimization.md), [optimize_latency.py](baseline/optimize_latency.py) |
| 팀원의 Layer-wise Drop 범위·절차 | [layerwise_drop.md](baseline/layerwise_drop.md) |
| 이전 H1 진단·4090 실험 / 베이스라인 | [h1_diagnosis.md](baseline/h1_diagnosis.md), [rtx4090_reproduction.md](baseline/rtx4090_reproduction.md), [docs/baseline.md](docs/baseline.md) |

GPU 최적화와 Layer-wise Drop 자료는 다른 브랜치에서 가져온 참고 자료입니다. 이 브랜치의 부분 학습 정확도 개선으로 합산하지 않습니다. 모든 학습 결과는 단일 seed이며, 이미지별 paired 구간과 학습 seed 간 변동성은 다릅니다.

<details>
<summary>설치·데이터 준비·실험 재현</summary>

### 설치

저장소 루트에서 실행합니다. Linux와 CUDA GPU, ImageNet 접근 권한이 필요합니다. 모델의 `timm`은 PyPI 버전 대신 공식 저장소 안의 custom 버전을 사용합니다.

```bash
git clone -b experiment/drop-ft https://github.com/hee4040/OnDevice.git
cd OnDevice

# 실험에 사용한 공식 코드 revision을 먼저 고정
git clone https://github.com/ds-kiel/ThinkingViT.git third_party/ThinkingViT
git -C third_party/ThinkingViT checkout 9254c4950d3159937ad682592d58a2be69708c42

bash scripts/bootstrap.sh
.venv/bin/python -m pip install -r requirements.txt
```

### 데이터

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/huggingface-cli login
.venv/bin/python scripts/prepare_imagenet.py --splits validation
.venv/bin/python scripts/prepare_imagenet.py --splits train --discard-shards
```

ImageNet 이용 조건에 동의하고 접근 승인받은 계정이 필요합니다. `--discard-shards`는 추출·manifest 저장 후 전용 임시 Parquet 캐시를 제거합니다. 원본 이미지는 유지하며, 같은 명령으로 준비를 재개할 수 있습니다.

### 평가와 전체 실험

```bash
# 원본·학습 전 Drop 평가
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate_drop.py

# 첫 실험 묶음: 원본 평가 폴더를 지정
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/run_drop_ft_study.py \
  --official-eval results/<원본_평가_폴더> \
  --output-dir results/<새_실험_폴더>

# 부분 학습 묶음: 완료된 KD 결과를 대조군으로 사용
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/run_partial_ft_study.py \
  --output-dir results/<새_부분학습_폴더>

# 부분 학습 checkpoint의 구조는 자동으로 감지
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate_drop.py \
  --weights results/<실험_폴더>/<방법>/train/student.pth

# 단위 테스트
.venv/bin/python -m unittest discover -s tests -v
```

부분 학습 실행기는 현재 원본·KD 대조군을 로컬 경로 `results/20261106T124010.390968Z-drop-eval-official/`와 `results/drop-ft-20261106/kd/eval/`에서 읽습니다. 새 PC에서 전체 실행기를 사용할 때는 이 원자료를 같은 위치에 준비하거나 실행기의 참조 경로를 조정해야 합니다. 개별 학습·평가는 위 공통 스크립트의 옵션으로 실행할 수 있습니다.

### 가중치와 원자료

| 로컬 경로 | 내용 |
| --- | --- |
| `checkpoints/thinkingvit_3h6h_ema.pth` | 공식 사전학습 EMA |
| `results/drop-ft-20261106/<방법>/train/student.pth` | KD·CE·Full·Feature 학습 가중치 |
| `results/partial-ft-20261109/<방법>/train/student.pth` | P·P+LN·P+LN+LoRA 학습 가중치 |
| 각 `eval/summary.json`, `eval/per_image.pt` | 정확도 요약·이미지별 원자료 |
| 각 `study.json`, `runner.log` | 실행 환경·명령·상태·소요 시간 |

`data/`, `checkpoints/`, `results/`, `third_party/`, `.venv/`, `.cache/`는 Git에 포함하지 않습니다. 이 README와 결과 MD에는 요약을 남기고, 가중치·이미지별 원자료는 필요할 때 별도로 공유합니다. 이전 H1의 상세 재현은 [h1_diagnosis.md](baseline/h1_diagnosis.md#재현), 설치 파일 설명은 [베이스라인 문서](docs/baseline.md)를 참고하세요.

</details>

## 참고 자료

[ThinkingViT 공식 코드](https://github.com/ds-kiel/ThinkingViT) · [논문](https://arxiv.org/abs/2507.10800) · [공식 EMA 가중치](https://huggingface.co/NCPS/thinkingvit_deit-3h-6h-imagenet1k) · [ImageNet-1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
