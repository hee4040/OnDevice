# OnDevice: ThinkingViT 선택적 토큰 재계산(H1) 실험

온디바이스 AI 4조 프로젝트 저장소입니다. ThinkingViT(3H → 6H)에서 6H가 일부 토큰만 다시 계산하게 만들어 연산을 줄이는 방법(H1)을 실험합니다.

> **이 브랜치(`experiment/h1-diagnosis`)는 장준용의 실험 브랜치입니다.**
> - main(준철님의 베이스라인)과 희우님의 H1 두 브랜치 결과를 RTX 4090에서 재현했습니다.
> - 그 위에 H1 실패 원인 분석, 토큰 제거, 파인튜닝, 추론 시간 측정을 더했습니다.
> - main과 다른 브랜치는 수정하지 않았습니다.

## 한눈에 보기

ImageNet-1K 검증 50,000장, 기준값 τ = 1.0 기준입니다. 추론 시간은 이미지 한 장의 평균이며, 검증 이미지 1,000장을 한 장씩(배치 1) 넣어 쟀습니다.

| 방법 | 전체 정확도 | 평균 GMACs | GPU (RTX 4090) | CPU 4코어 | CPU 1코어 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3H만 사용 | 73.53% | 1.25 | 4.0 ms | 11.3 ms | 28.0 ms |
| 원래 ThinkingViT (τ = 1.0) | 80.30% | 2.92 | 5.5 ms | 20.4 ms | 56.6 ms |
| 모든 이미지를 6H까지 | 81.44% | 5.87 | 8.7 ms | 38.6 ms | 114.0 ms |
| 기존 H1 방식 50% | 57.77% | 2.23\* | 5.7 ms | 20.6 ms | 57.0 ms |
| **토큰 제거 25% + 파인튜닝** | **78.43%** | **1.67** | 5.7 ms | 15.7 ms | **38.2 ms** |
| **토큰 제거 50% + 파인튜닝** | **79.65%** | **2.07** | 5.7 ms | 17.1 ms | **44.1 ms** |

\* 이론값입니다. 지금 구현은 모든 토큰을 계산한 뒤 결과를 버리므로 실제 시간은 원래 모델과 같습니다.

**기존 H1이 망가진 이유**
- 계산하지 않은 토큰이 처음 상태 그대로 남아, 다른 토큰의 계산을 방해했습니다.
- 이 토큰을 6H에서 아예 빼면(토큰 제거) 학습 없이도 정확도가 대부분 돌아옵니다.

**토큰 제거 + 파인튜닝의 성능**
- 같은 연산량에서 τ만 조정한 원래 모델보다 정확도가 1.7~2.4%p 높습니다.
- 원래 모델(τ = 1.0) 그대로와 비교하면 정확도는 0.65~1.87%p 낮고, 연산량은 29~43% 적습니다.

**실제 속도**
- CPU에서는 줄인 연산만큼 빨라집니다. 1코어 기준 56.6ms → 38.2ms입니다.
- GPU에서 한 장씩 처리할 때는 빨라지지 않습니다.

## 문서 지도

| 알고 싶은 것 | 볼 문서 |
| --- | --- |
| H1 실험 전체를 쉬운 말로 (A4 3쪽) | [`docs/H1_실험보고서_1103.pdf`](docs/H1_실험보고서_1103.pdf) (한글 원본 `.hwp`도 같은 폴더) |
| H1 실험 상세: 실험 1~7, 표, 재현 명령 | [`baseline/h1_diagnosis.md`](baseline/h1_diagnosis.md) |
| RTX 4090 재현과 H1 76~100% sweep | [`baseline/rtx4090_reproduction.md`](baseline/rtx4090_reproduction.md) |
| 베이스라인(early exit) 측정, RTX 3090 | [`docs/baseline.md`](docs/baseline.md) |
| ThinkingViT가 틀리는 사례 분석 | [`baseline/failure_analysis.md`](baseline/failure_analysis.md) |
| early-exit 실행기 사용법 | [`baseline/README.md`](baseline/README.md) |
| 결과 파일(JSON·CSV·그림) 날짜별 안내 | [`baseline/reports/README.md`](baseline/reports/README.md) |

## 1. 필요한 환경

| 항목 | 실험에 쓴 환경 | 필요 조건 |
| --- | --- | --- |
| OS | Ubuntu 22.04.5 LTS | 리눅스 (Windows는 확인하지 않음) |
| Python | 3.10.12 | 3.10 |
| GPU | RTX 4090 24GB (베이스라인은 RTX 3090에서도 확인) | 평가 1GB 이하, 파인튜닝(배치 128) 약 6GB |
| NVIDIA 드라이버 | 535.183.01 | 525.60 이상. CUDA 12.4 런타임은 pip가 함께 설치하므로 CUDA Toolkit은 필요 없음 |
| CPU / RAM | Core i5-13400 (10코어 16스레드) / 32GB | 파인튜닝 속도는 CPU의 이미지 디코딩에 좌우됨 (데이터 로더 14개 기준) |
| 디스크 | 약 155GB 사용 | 검증 6.4GB, train 140GB, `.venv` 5.4GB, 가중치 85MB. train의 원본 Parquet를 지우지 않으면 약 140GB가 더 필요 |
| 계정 | Hugging Face | ImageNet-1K 접근 승인을 받은 계정 |

## 2. 설치

```bash
git clone -b experiment/h1-diagnosis https://github.com/hee4040/OnDevice.git
cd OnDevice
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt   # 실험에 쓴 패키지 46개, 버전 고정
bash scripts/bootstrap.sh                    # ThinkingViT 공식 코드 clone, 가중치 다운로드·검증
```

| 파일 | 들어 있는 것 | 언제 쓰나 |
| --- | --- | --- |
| [`requirements.txt`](requirements.txt) | 실험 PC의 `pip freeze` 그대로 46개 (torch 2.5.1, torchvision 0.20.1 등) | 처음 설치할 때 이것 하나면 됨 |
| [`requirements-data.txt`](requirements-data.txt) | 데이터 준비용 3개 | `bootstrap.sh --data-only`가 설치 |
| [`requirements-analysis.txt`](requirements-analysis.txt) | 그림용 matplotlib | 그림만 다시 그릴 때 |
| `third_party/ThinkingViT/requirements.txt` | 모델 실행용 (torch 등) | `bootstrap.sh`가 설치. 버전은 `requirements.txt`와 같음 |

`timm`은 PyPI에서 설치하지 않습니다. 스크립트가 `third_party/ThinkingViT/` 안의 수정된 timm을 씁니다. ThinkingViT 공식 코드는 commit `9254c4950d3159937ad682592d58a2be69708c42`입니다.

## 3. 데이터 준비

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/huggingface-cli login                                   # ImageNet-1K 접근 승인된 계정의 읽기 토큰
.venv/bin/python scripts/prepare_imagenet.py --splits validation  # 5만 장 (평가용)
.venv/bin/python scripts/prepare_imagenet.py --splits train       # 128만 장 (파인튜닝용, 학교망 기준 약 24시간)
```

- 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 받습니다.
- 토큰은 코드나 커밋에 넣지 않습니다.
- 자세한 옵션은 [`docs/baseline.md`](docs/baseline.md)의 "3. 데이터 준비와 재개"에 있습니다.

## 4. 실험 재현 순서

모든 명령은 저장소 루트에서 `CUDA_VISIBLE_DEVICES=0 .venv/bin/python`으로 실행합니다. 시간은 RTX 4090 기준이고, 결과는 `results/<시각>-<이름>/`에 저장됩니다.

| 순서 | 할 일 | 스크립트 | 시간 |
| ---: | --- | --- | ---: |
| 1 | 베이스라인 정확도·지연 (공식 출력과 일치 확인) | `baseline/evaluate.py --mode accuracy` / `--mode latency` | 수 분 |
| 2 | H1 진단: 실험 1~4 | `baseline/evaluate_h1_diagnosis.py` → `baseline/plot_h1_diagnosis.py` | 15분 |
| 3 | 학습 없는 토큰 제거 평가 | `baseline/evaluate_drop.py` | 2분 |
| 4 | 토큰 제거 파인튜닝 (train 필요) | `baseline/finetune_drop.py --epochs 3 --lr 5e-5` | 50분 |
| 5 | 파인튜닝 모델 평가 | `baseline/evaluate_drop.py --weights results/<4번 폴더>/student.pth` | 2분 |
| 6 | 6H 단계 시간 / 방법별 전체 추론 시간 | `baseline/measure_latency.py` / `--mode pipeline --images 1000` | 2분 / 15분 |
| 7 | 분석과 그림 | `baseline/analyze_followup.py` (인자는 문서 참고) | 1분 |

전체 명령과 인자는 [`baseline/h1_diagnosis.md`의 "재현"](baseline/h1_diagnosis.md#재현)에 있습니다. 단위 테스트는 `.venv/bin/python -m unittest discover -s tests`로 돌리며, 26개가 통과합니다.

## 5. 폴더 구조

```text
OnDevice/
├── README.md                  # 이 문서
├── requirements.txt           # 패키지 전체 46개, 버전 고정
├── requirements-data.txt      # 데이터 준비용 (bootstrap.sh가 사용)
├── requirements-analysis.txt  # 그림용 matplotlib
├── docs/
│   ├── H1_실험보고서_1103.pdf    # 쉬운 요약 보고서 (한글 원본 .hwp 포함)
│   └── baseline.md              # 10/14 베이스라인 측정 상세 (원래 루트 README)
├── scripts/                   # 설치·데이터·가중치 준비, 공식 평가기 호출
├── baseline/                  # 실험 코드와 문서 (아래 표)
│   └── reports/<날짜>/         # 공유용 결과: JSON·CSV·그림
└── tests/                     # 단위 테스트
```

`.venv/`, `.cache/`, `data/`, `checkpoints/`, `third_party/`, `results/`는 실행하면서 생기며 Git에 올리지 않습니다. 파인튜닝 가중치(`student.pth`)와 이미지별 결과(`per_image.pt`)도 용량 때문에 올리지 않았습니다.

### `baseline/` 파일 안내

| 단계 | 파일 | 하는 일 | 작성 |
| --- | --- | --- | --- |
| 베이스라인 | `early_exit.py` | 3H가 확신하면 6H를 실제로 건너뛰는 실행기 | 준철, 10/14 |
| | `evaluate.py` | 공식 결과와 일치 확인, 정확도·지연 측정 | 준철, 10/14 |
| 실패 분석 | `analyze_failures.py`, `audit_failures.py`, `analyze_stages.py` | 3H·6H가 틀리는 사례, ReaL 라벨 재평가 | 준철, 10/15 |
| attention 분석 | `analyze_attention.py`, `occlude_attention.py` | CLS attention 지도, 패치 가림 실험 | 준철, 10/27 |
| H1 원본 | `selective_refinement.py` | 기존 H1 방식: 고른 토큰만 갱신, 나머지는 그대로 | 희우 (`experiment/h1-selective-token-refinement`에서 가져옴) |
| | `mlp_selective_refinement.py` | attention은 전체, MLP만 고른 토큰에 적용 | 희우 (`experiment/h1-attention-full-mlp-selective`에서 가져옴) |
| 4090 재현 | `plot_h1_ratio_sweep.py` | H1 76~100% sweep 그림 | 장준용, 11/1 |
| H1 진단 | `evaluate_h1_diagnosis.py`, `plot_h1_diagnosis.py` | 실험 1~4: 기존 H1 vs 토큰 제거, 앞 k층 전체 계산, 같은 연산량 비교, sink 토큰 | 장준용, 11/2 |
| 후속 실험 | `evaluate_drop.py` | 이미지마다 3H·6H·토큰 제거의 정답 여부 기록 (원래·파인튜닝 모델) | 장준용, 11/3 |
| | `finetune_drop.py` | 토큰 제거 파인튜닝 (원래 모델을 교사로 쓰는 지식 증류) | 장준용, 11/3 |
| | `measure_latency.py` | 6H 단계 시간, 방법별 전체 추론 시간 (GPU, CPU 4코어, CPU 1코어) | 장준용, 11/3 |
| | `analyze_followup.py` | 같은 연산량 비교, 부트스트랩, 반 나눠 검증, 그림 | 장준용, 11/3 |

## 6. 브랜치

| 브랜치 | 내용 |
| --- | --- |
| `main` | 베이스라인: early exit, 실패 분석, attention 분석 (준철, 10/14~10/27) |
| `experiment/h1-selective-token-refinement` | 기존 H1 방식 첫 실험 (희우, 10/21) |
| `experiment/h1-attention-full-mlp-selective` | MLP만 고른 토큰에 적용하는 방식 (희우, 10/22) |
| `experiment/rtx4090-reproduction` | RTX 4090 재현과 76~100% sweep (장준용, 11/1) |
| `experiment/h1-diagnosis` | **이 브랜치.** 위 내용 전부에 H1 진단, 토큰 제거, 파인튜닝, 추론 시간을 더함 (장준용, 11/2~11/3) |

## 참고 자료

- [ThinkingViT 공식 코드](https://github.com/ds-kiel/ThinkingViT) · [논문](https://arxiv.org/abs/2507.10800) · [3H → 6H EMA 가중치](https://huggingface.co/NCPS/thinkingvit_deit-3h-6h-imagenet1k)
- [ImageNet-1K 데이터셋](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
