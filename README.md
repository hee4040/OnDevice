# OnDevice: ThinkingViT ImageNet-1K Baseline

**Selective Token Refinement for Efficient ThinkingViT** 수업 프로젝트의 베이스라인 실험 코드다. 공식 ThinkingViT의 DeiT 기반 **3H → 6H** 모델을 ImageNet-1K에서 평가하고, 이후 선택적 토큰 갱신 모델과 비교할 기준을 마련한다.

현재 제공하는 기능은 데이터 준비, 공식 EMA 가중치 다운로드, 공식 평가 재현, 실제 early-exit baseline, 정확도·batch-1 지연시간·GPU 메모리 측정이다. 선택적 토큰 갱신과 fine-tuning은 아직 구현하지 않았다.

## 베이스라인 실측 결과 — 2026-10-14

추가 분석: [ThinkingViT 실패 분석 — 수치·예시 이미지·ReaL 재평가](baseline/failure_analysis.md) (2026-10-15).

**같은 배치에서 공식 forward와 실제 early exit를 비교했을 때, threshold별 validation 50,000장 전체의 Top-1 예측과 분기 선택이 모두 일치했다.** threshold 1.0의 batch-1 평균 latency는 19.277ms에서 12.371ms로 약 **35.8% 감소**했다. 이는 RTX 3090에서 수행한 baseline 측정이며, 선택적 토큰 갱신의 성능이나 모바일 기기 실측 결과는 아니다.

### 전체 validation 정확도

GPU 1장, batch 128, FP32·AMP off·TF32 허용 조건에서 같은 입력 배치로 직접 비교했다. 공식 경로는 6H를 항상 실행하고, early exit는 배치 안에서 어려운 이미지만 6H로 보낸다.

| Threshold | 공식 Top-1 (%) | Early exit Top-1 (%) | 공식 Top-5 (%) | Early exit Top-5 (%) | Early exit 실제 6H 실행 비율 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10.0 | 73.534 | 73.534 | 92.048 | 92.048 | 0.000% |
| 0.0 | 81.440 | 81.440 | 95.720 | 95.720 | 100.000% |
| 1.0 | 80.304 | 80.304 | 95.278 | 95.278 | 36.008% |

모든 threshold에서 Top-1 예측 불일치와 분기 선택 불일치는 각각 **0 / 50,000**이었다. threshold 10.0과 0.0의 logits는 정확히 일치했다. threshold 1.0은 logits의 최대 절댓값 차이가 **0.00828648**이므로 완전한 수치 일치로 보고하지 않는다. 전체 데이터에서 확인한 것은 Top-1 예측·분기 선택의 일치와 위 정확도이며, 모든 logits가 사전 검증의 허용 오차를 만족한다고 주장하지 않는다.

이전 공식 실행기(batch 512)의 Top-1은 73.536 / 81.440 / 80.310%였다. batch 128의 공식 경로에서도 소수의 정답 수 차이가 나타났으므로, 이전 수치와의 차이를 early exit에 의한 정확도 손실로 해석하지 않는다. 배치 구성과 GPU 수치 연산 조건을 맞춘 직접 비교 결과를 기준으로 사용한다.

[전체 정확도 비교 JSON](baseline/reports/2026-10-14/accuracy.json) · [기존 공식 실행기 결과](baseline/reports/2026-10-14/official_accuracy.json)

### Batch-1 latency

같은 GPU에서 같은 validation 표본 1,000장(seed 42)을 사용했다. 각 경로의 두 분기를 각각 50회 warmup했고, CUDA 동기화 wall-clock 시간으로 측정했다. Entropy 판단과 분기 동기화는 포함하고 이미지 로딩·전처리·GPU 전송은 제외했다.

| Threshold | 공식 평균 (ms) | Early exit 평균 (ms) | Early exit p95 (ms) | 표본 내 실제 6H 실행 비율 |
| --- | ---: | ---: | ---: | ---: |
| 10.0 | 19.388 | 9.114 | 9.599 | 0.0% |
| 0.0 | 19.240 | 19.279 | 19.851 | 100.0% |
| 1.0 | 19.277 | 12.371 | 19.443 | 33.2% |

모든 이미지를 6H로 보내는 threshold 0.0에서는 두 경로의 시간이 비슷하다. threshold 1.0의 peak allocated memory는 두 경로 모두 97.695 MiB였다. 정확도는 전체 50,000장, latency는 1,000장 표본이므로 두 실험의 6H 실행 비율은 다르다. 한 번의 실행 결과이며 반복 실험 신뢰구간은 측정하지 않았다.

[Latency·메모리 상세 JSON](baseline/reports/2026-10-14/latency.json) · [실험 환경·revision·해시](baseline/reports/2026-10-14/provenance.json) · [측정 표본 인덱스](baseline/reports/2026-10-14/latency_sample_indices.json)

### 측정 환경과 재현

- NVIDIA GeForce RTX 3090 1장, Python 3.10.12, PyTorch 2.5.1+cu124 / CUDA 12.4
- 공식 ThinkingViT commit: `9254c4950d3159937ad682592d58a2be69708c42`, 원본 코드 변경 없음
- 공식 일반 EMA 가중치, 입력 224×224, crop_pct 0.9, bicubic 보간, ImageNet mean/std
- 정확도는 GPU 한 장에 큰 배치를 사용하고, latency는 단일 이미지의 응답시간을 보기 위해 batch 1을 사용한다.

아래 설치·데이터 준비를 마친 뒤 저장소 루트에서 실행한다.

```bash
# 전체 50,000장 정확도 및 동일 배치의 공식 출력과 직접 비교
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py \
  --mode accuracy --batch-size 128 --verify-samples 128 --compare-official

# 같은 1,000장에서 공식 forward와 실제 early exit의 latency 비교
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py \
  --mode latency --samples 1000 --warmup 50 --verify-samples 100
```

공식 실행기와 실제 early-exit 코드는 분리되어 있다. [`scripts/baseline.py`](scripts/baseline.py)는 공식 평가를 호출하고, [`baseline/`](baseline/README.md)은 실제 early exit와 검증·측정을 담당한다. 공유용 요약은 [`baseline/reports/2026-10-14/`](baseline/reports/2026-10-14/)에, 이미지별 전체 측정 로그는 로컬 `results/`에 저장한다.

## 1. 요구 환경

- Linux, Python 3.10, Git
- 데이터·가중치 다운로드가 가능한 네트워크
- ImageNet-1K 접근 승인을 받은 Hugging Face 계정
- 평가 시 CUDA를 사용할 수 있는 NVIDIA GPU와 드라이버
- train 전체를 준비할 경우 수백 GB의 디스크 여유: 원본 Parquet와 추출 이미지를 함께 보관한다.

GPU는 데이터와 가중치를 받을 때는 필요하지 않다. 이 저장소의 평가 스크립트는 CUDA를 요구한다. 모델 의존성은 [공식 requirements.txt](https://github.com/ds-kiel/ThinkingViT/blob/main/requirements.txt)를 따르며, 현재 PyTorch 2.5.1 / torchvision 0.20.1을 사용한다. 공식 저장소에 포함된 custom `timm`을 사용해야 한다.

## 2. 빠른 시작

### 저장소와 가상환경 준비

```bash
git clone https://github.com/hee4040/OnDevice.git
cd OnDevice
bash scripts/bootstrap.sh --data-only
```

프로젝트 내부 `.venv`에 데이터 준비용 패키지를 설치한다. 이후 명령은 저장소 루트에서 실행한다.

### ImageNet 접근 승인 및 로그인

먼저 [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)에서 이용 조건을 확인하고 접근 승인을 받는다. 그 계정의 읽기 권한 토큰으로 터미널에서 로그인한다.

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/huggingface-cli login
```

토큰을 코드나 커밋에 넣지 않는다. 로그인 캐시가 있는 `.cache/`는 Git 추적에서 제외한다.

### Validation 준비

초기 베이스라인 평가에는 validation만 있으면 된다.

```bash
.venv/bin/python scripts/prepare_imagenet.py --splits validation
```

완료되면 `data/imagenet1k/validation/`에 이미지 50,000장과 클래스 폴더 1,000개가 생성된다.

### 공식 모델과 가중치 설치

```bash
bash scripts/bootstrap.sh
```

이 명령은 다음 작업을 수행한다.

1. 공식 ThinkingViT 코드를 `third_party/ThinkingViT/`에 clone한다. 이미 있으면 그 checkout을 유지한다.
2. 공식 모델 의존성을 `.venv`에 설치한다.
3. [ThinkingViT DeiT 3H → 6H의 일반 EMA 가중치](https://huggingface.co/NCPS/thinkingvit_deit-3h-6h-imagenet1k)를 다운로드한다.
4. 가중치를 모델에 `strict=True`로 로딩하여 확인하고, 공식 평가기용 checkpoint로 저장한다.

가중치 위치는 `checkpoints/thinkingvit_3h6h_ema.pth`다. 같은 폴더의 JSON에 모델 저장소, revision, SHA256을 기록한다. 일반 모델을 사용하며 800-epoch 모델은 별도 실험으로 취급한다.

### 정확도 및 지연시간 평가

```bash
# Validation 전체 50,000장: Top-1, Top-5, 단계별 출력 선택 비율
.venv/bin/python scripts/baseline.py --mode accuracy

# Validation에서 1,000장 표본: batch-1 latency, peak GPU memory
.venv/bin/python scripts/baseline.py --mode latency

# 실제 early exit: 공식 경로와 같은 이미지로 batch-1 latency 비교
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py --mode latency

# 실제 early exit: 전체 정확도와 같은 배치의 공식 예측 비교
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py \
  --mode accuracy --batch-size 128 --compare-official
```

실행마다 `results/` 아래에 UTC timestamp가 포함된 결과 폴더가 생성된다. 초기 베이스라인은 threshold `10.0`, `0.0`, `1.0`을 비교한다. 정확도와 지연시간의 해석은 아래 **측정 시 주의점**을 함께 확인한다. early-exit 실행기의 세부 사용법은 [`baseline/README.md`](baseline/README.md)에 정리했다.

## 3. 데이터 준비와 재개

| Split | 이미지 수 | 사용 목적 |
| --- | ---: | --- |
| Train | 1,281,167 | 후속 학습·fine-tuning |
| Validation | 50,000 | 정답이 공개된 베이스라인 평가 |
| Test | 100,000 | 정답 비공개, 서버 제출 평가 |

ThinkingViT 공식 성능 비교에는 validation을 사용한다. 이 저장소의 다운로드 스크립트는 train과 validation을 지원하며 test는 받지 않는다. [데이터셋 설명](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

```bash
# Validation 이후 train 전체 준비: 기본 실행 순서
.venv/bin/python scripts/prepare_imagenet.py

# Train만 추가 준비
.venv/bin/python scripts/prepare_imagenet.py --splits train

# 다른 디스크에 저장
.venv/bin/python scripts/prepare_imagenet.py --output /path/to/imagenet1k
```

중단 후 같은 명령을 다시 실행하면 완료한 shard를 건너뛰고 이어서 준비한다. 원본 이미지 bytes를 재인코딩하거나 크기를 변경하지 않고 저장하며, 최초 추출 시 이미지를 디코딩하여 확인한다.

느린 연결에 대비해 메타데이터 요청 timeout은 60초, 다운로드 응답 timeout은 120초로 설정한다. 일시적인 연결 오류와 HTTP 408·429·5xx 오류에는 최대 8회 추가 시도하며, 재시도 간 대기는 5초부터 최대 60초까지 늘어난다. 인증·권한 오류는 즉시 표시한다. 재시도를 모두 소진해 종료되어도 완료 기록과 다운로드 캐시는 유지된다. 기존 `HF_HUB_ETAG_TIMEOUT`, `HF_HUB_DOWNLOAD_TIMEOUT` 환경변수가 있으면 해당 값을 우선한다.

```bash
# 연결 지연이 길 때 timeout과 재시도 횟수를 직접 지정
HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=180 \
  .venv/bin/python scripts/prepare_imagenet.py --splits train --retries 12
```

클래스 폴더 `0000`~`0999`는 Hugging Face의 정수 label을 보존한다. 폴더의 사전순 정렬과 분류 인덱스가 일치한다. 원본 파일명, 이미지 크기(bytes), SHA256은 `.preparation/`의 shard별 manifest에 남긴다. 재개 시 완료 여부는 파일 존재와 크기로 확인하며 전체 SHA256을 다시 계산하지는 않는다.

데이터 revision은 최초 접근 성공 시 고정한다. split별 `*-complete.json`은 전체 이미지 수와 클래스별 개수 검증 후에 생성된다. validation은 클래스마다 50장이어야 한다. 원본 Parquet는 `.cache/huggingface/`에 남겨 재개에 사용하며 자동 삭제하지 않는다.

## 4. 평가 설정과 실행 예시

실제로 6H 실행을 생략하는 baseline은 별도 [`baseline/`](baseline/README.md)에 있다. 아래 기존 스크립트는 공식 forward를 그대로 평가하며, `baseline/evaluate.py`는 공식 경로와 실제 early exit의 출력 일치 및 속도를 비교한다. 정확도는 batch 128, 지연시간은 batch 1을 기본으로 사용한다.

| 항목 | 기본 설정 |
| --- | --- |
| 모델 / stages | `thinkingvit` / `3 6` |
| 입력 / 패치 / Transformer blocks | 224×224 / 16×16 / 12 |
| Embedding dimension | 192 → 384 |
| 가중치 | 공식 일반 EMA export |
| 정확도 batch size | 공식 실행기 512 / early-exit 실행기 128 |
| 전처리 | 공식 모델 config 자동 적용 |
| 정밀도 | float32, AMP off, 공식 코드와 같이 TF32 허용 |
| 기본 thresholds | 10.0 / 0.0 / 1.0 |

현재 공식 모델 config는 crop_pct=0.9, bicubic 보간, ImageNet mean/std를 사용한다. 학습 설정과 모델 파라미터는 변경하지 않는다.

```bash
# GPU 메모리가 부족하면 batch size를 명시적으로 조절
.venv/bin/python scripts/baseline.py --mode accuracy --batch-size 128

# 공식 threshold sweep
.venv/bin/python scripts/baseline.py --mode accuracy \
  --thresholds 0 0.1 0.2 0.3 0.5 0.8 1.0 1.2 1.4 1.6 2 5

# 데이터 경로 지정: 그 아래 validation/이 있어야 함
.venv/bin/python scripts/baseline.py --mode accuracy \
  --data-dir /path/to/imagenet1k

# Latency 표본 수와 warmup 변경
.venv/bin/python scripts/baseline.py --mode latency \
  --samples 1000 --warmup 50 --seed 42
```

`--checkpoint`와 `--repo`로 다른 checkpoint 및 공식 코드 경로를 지정할 수도 있다. 데이터 개수 검사기는 이 저장소에서 생성한 숫자 클래스 폴더 구성을 요구한다. 일반 synset 폴더 형태의 기존 ImageNet은 그대로 이 검사기를 통과하지 않는다. batch size는 자동으로 축소하지 않는다.

### 공식 참고 성능

| 조건 | Threshold | Top-1 (%) | GMACs |
| --- | ---: | ---: | ---: |
| 3H 출력 | 10.0 | 73.536 | 1.250 |
| 전체 3H → 6H 출력 | 0.0 | 81.440 | 5.850 |
| Entropy 기반 출력 선택 | 1.0 | 80.310 | 2.907 |

위 값은 [공식 README](https://github.com/ds-kiel/ThinkingViT)의 보고값이며, 이 프로젝트의 공식 실행기(batch 512) 평가에서도 같은 Top-1을 재현했다. 별도 early-exit 실행기의 결과와 측정 조건은 이 문서 상단의 실측 결과에 정리한다.

### 결과 파일

- 공통: `environment.json`, `summary.json`, `summary.csv`
- 정확도: threshold별 공식 실행 로그와 명령 인수 JSON
- 지연시간: `preprocessing.json`, `sample_indices.json`, threshold별 개별 지연시간과 선택 단계 JSON

실행 환경에는 GPU, PyTorch/CUDA 버전, 공식 코드 commit과 변경 상태, checkpoint SHA256을 기록한다. 설치 시의 코드 commit은 `third_party/ThinkingViT.commit`, 설치 패키지는 `requirements-installed.txt`에도 남긴다. 로컬 환경 기록과 측정 결과는 기본 Git 추적에서 제외한다.

## 5. 측정 시 주의점

### 출력 선택과 실제 early-exit 구분

확인한 [공식 forward](https://github.com/ds-kiel/ThinkingViT/blob/9254c4950d3159937ad682592d58a2be69708c42/timm/models/vision_transformer.py)는 3H와 6H를 모두 계산한 뒤 entropy로 logits를 선택한다. 따라서 threshold=10의 결과는 **3H 출력의 정확도**지만, 그 forward의 실행시간은 3H만 계산한 시간이 아니다.

Entropy는 전체 1,000개가 아닌 **상위 10개 클래스 확률을 재정규화하여 계산**한다. 향후 선택적 갱신 모델과 비교할 때도 같은 정의를 적용해야 한다.

[공식 평가 코드](https://github.com/ds-kiel/ThinkingViT/blob/9254c4950d3159937ad682592d58a2be69708c42/validate.py)의 GMACs는 `1.25 + 4.6 × stage_2_fraction`이다. 이는 routing을 가정한 추정치이며, profiler로 측정한 실행 연산량이 아니다. 결과 파일에서도 `gmacs_official_routing_estimate`로 구분한다.

`baseline/early_exit.py`는 같은 Entropy 규칙과 Token Recycling을 유지하면서 실제 실행을 분기한다. batch-1 경로는 확실한 이미지에서 즉시 반환하고, 정확도용 배치 경로는 어려운 이미지만 모아 6H를 실행한 뒤 원래 순서로 출력을 합친다. 공식 모델 코드와 가중치는 변경하지 않는다. 후속 선택적 토큰 갱신 모델의 속도는 이 early-exit 기준선과 비교한다.

### Latency 및 메모리 측정 범위

Latency는 이 프로젝트에서 추가한 프로토콜이다. batch=1, seed=42로 validation 전체에서 1,000장을 무작위 선택한다. 기존 공식 실행기는 threshold마다 50회 warmup하고, early-exit 비교 실행기는 각 경로를 측정하기 전 두 분기를 각각 50회 warmup한다. CUDA 동기화 wall-clock 시간으로 평균, p50, p95를 기록한다.

이미지 로딩·전처리·CPU→GPU 전송은 제외하고 모델 forward의 Python dispatch 비용을 포함한다. peak allocated/reserved GPU memory에는 모델이 포함되며 시스템 RAM 사용량과 다르다. 데스크톱 GPU에서 측정한 값을 온디바이스 실측값으로 표기하지 않는다.

early exit의 GPU → CPU 분기 동기화도 시간에 포함한다. `stage_2_fraction`은 6H 출력 선택 비율이며, 별도 `stage_2_execution_fraction`이 실제 실행 비율이다. 공식 경로는 항상 6H를 실행한다. 전체 측정 중 일부 이미지라도 6H로 진입하면 early exit도 같은 peak memory를 보일 수 있다. 한 번의 측정 결과이므로 반복 실험의 신뢰구간은 제공하지 않는다.

## 6. 저장소 구조와 검증

```text
OnDevice/
├── README.md
├── requirements-data.txt
├── baseline/
│   ├── README.md               # 실제 early-exit 사용법과 검증 방법
│   ├── early_exit.py           # batch-1 / 배치 내 선택적 6H 실행
│   ├── evaluate.py             # 출력 검증, 정확도 및 지연시간 비교
│   └── reports/                # 공유용 결과 요약과 재현 정보
├── scripts/
│   ├── bootstrap.sh            # 가상환경, 공식 코드·의존성, 가중치 설치
│   ├── prepare_imagenet.py     # 다운로드, ImageFolder 변환, 개수 검증
│   ├── prepare_weights.py      # 공식 EMA 다운로드 및 strict loading
│   └── baseline.py             # 공식 정확도 평가 및 batch-1 측정
└── tests/
    ├── test_preparation.py     # 결과 파싱, 불완전 데이터 거부 검사
    ├── test_download_retry.py  # timeout 재시도, 영구 오류, 재시도 한도 검사
    └── test_early_exit.py      # 실제 실행 생략, 특징 재활용, 배치 복원 검사
```

`.venv/`, `.cache/`, `data/`, `checkpoints/`, `third_party/`, `results/`는 실행 중 생성하며 커밋에 포함하지 않는다.

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 -m compileall -q scripts baseline tests
bash -n scripts/bootstrap.sh
```

2026-10-14 기준 validation 50,000장·1,000개 클래스 준비와 개수 검증을 완료했으며, 단위 테스트 16개를 통과했다. GPU에서 수행한 전체 정확도 평가 및 공식 출력과의 비교 결과는 상단에 별도로 기록한다. 단위 테스트만으로 모델 정확도를 검증했다고 간주하지 않는다.

## 참고 자료

- [ThinkingViT 공식 코드](https://github.com/ds-kiel/ThinkingViT)
- [ThinkingViT 논문](https://arxiv.org/abs/2507.10800)
- [공식 3H → 6H EMA 가중치](https://huggingface.co/NCPS/thinkingvit_deit-3h-6h-imagenet1k)
- [ImageNet-1K 데이터셋](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
