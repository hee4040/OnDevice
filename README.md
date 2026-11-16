# Efficient ThinkingViT — 최종 통합 결과

`integration/efficient-kd` 브랜치는 **기존 전체 KD 학습**, **종료 게이트·토큰 유지율 공동 조정**, **중복 임베딩 재사용**, **고정 폭 CUDA Graph 실행**을 통합한다.
정확도만 높이는 것이 아니라 **정확도–평균 연산량–실제 지연시간**을 함께 비교한다.

**주 비교 기준은 현재 `main`의 Drop 이전 ThinkingViT다.** 공식 EMA 가중치, 3H→6H, 토큰 전체 유지, entropy τ=1의 실제 early-exit 경로를 사용한다.
이번 GPU에서도 원본 정확도 **80.304%**와 6H 진입 **18,004/50,000장**을 재현했다. KD 79.67%는 중간 모델의 보조 대조군이며 원본 기준을 대신하지 않는다.

[상세 측정 결과](reports/RESULTS.md) · [구현·비교 방법](docs/method.md) · [main 베이스라인 근거](reports/original-main-protocol.json)

## 1. 최종 결과: main 원본 대비

ImageNet-1K validation **50,000장**, FP32·TF32 허용. GMACs는 이미지당 평균이다.
시간은 같은 RTX 3070 Ti Laptop GPU에서 같은 1,000장을 batch 1로 처리한 **3회 반복 평균의 중앙값**이다.

| 모델 / 모드 | Top-1 | 원본 대비 정확도 | 평균 GMACs | 원본 대비 연산량 감소 | 이미지당 시간 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **main 원본 ThinkingViT** | **80.304%** | 기준 | **2.9438** | 기준 | **10.032ms** |
| 같은 원본 + 실행 최적화만 | 80.304% | +0.000%p | 2.9230 | 0.71% | 2.149ms |
| 통합 절약 `economy` | 78.388% | −1.916%p | 1.6801 | **42.93%** | **1.759ms** |
| **통합 기본 `balanced`** | **80.018%** | **−0.286%p** | **2.0714** | **29.63%** | **1.909ms** |
| 통합 정확도 우선 `accuracy` | **80.438%** | **+0.134%p** | **2.2651** | **23.05%** | **2.038ms** |

- **기본 모드:** 원본 대비 정확도 0.286%p를 내주고 연산량을 29.63% 줄였다.
- **정확도 우선:** 이번 평가에서는 원본보다 정확도가 0.134%p 높으면서 연산량은 23.05% 적었다.
- **절약 모드:** 정확도 손실을 더 허용해 연산량을 42.93% 줄였다.

시간 단축에는 **정책 변경과 실행 최적화의 효과가 함께** 들어 있다. 알고리즘만으로 약 5배 가속됐다고 해석하지 않는다.
같은 원본 가중치·전체 토큰·τ=1을 유지한 실행 최적화 대조군은 **2.149ms**다.
동일한 실행 최적화 조건에서 비교하면 통합 기본은 **1.909ms**, 정확도 우선은 **2.038ms**다.
원본에 실행 최적화만 적용했을 때는 5만 장의 예측·정오답·분기 선택이 모두 원본과 일치했다.

`main`의 과거 RTX 3090 시간은 이 표에 재사용하지 않았다. 현재 장치에서 원본과 통합 모델을 다시 측정했다.
현재 정확도 배치는 64이며 `main`의 과거 측정은 128이다. 모델 가중치·공식 코드 revision·전처리·gate 조건은 맞추고 원본 정확도를 재현했다.

## 2. 어떤 모드를 쓰면 되는가

| 모드 | 선택 기준 | 실제 정책 |
| --- | --- | --- |
| `economy` | 평균 연산량 절감 우선 | entropy < 1이면 종료, 나머지는 25% 유지 |
| `balanced` | 기본 정확도–연산량 균형 | entropy < 0.8이면 종료, 0.8 이상 1.25 미만은 25%, 1.25 이상은 50% 유지 |
| `accuracy` | 정확도를 더 확보 | entropy < 0.75이면 종료, 1.15 미만은 25%, 1.75 미만은 50%, 그 이상은 75% 유지 |

유지율은 **버리는 비율이 아니라 남기는 패치 비율**이다. 원래 패치 196개와 별도로 CLS는 항상 남긴다.

| 패치 유지율 | 유지 패치 | CLS 포함 6H 입력 |
| --- | ---: | ---: |
| 25% | 49 | 50토큰 |
| 50% | 98 | 99토큰 |
| 75% | 147 | 148토큰 |
| 원본 100% | 196 | 197토큰 |

`kd50`과 `full`도 제공하지만 보조 비교용이다. 둘 다 **KD 학생 가중치**를 사용한다.
특히 기본 명령의 `full`은 **main의 공식 원본이 아니다**. 원본 비교에는 반드시 `--official --runtime native --policies full`을 사용한다.

## 3. FP16은 별도 선택 사항

아래는 FP16으로 다시 평가한 5만 장 정확도와 Graph 지연시간이다. FP32 정확도와 섞어서 보고하지 않는다.

| 모드 | FP16 Top-1 | 해당 모드 FP32 대비 | 평균 GMACs | Graph 시간 |
| --- | ---: | ---: | ---: | ---: |
| `economy` | 78.380% | −0.008%p | 1.6801 | 1.195ms |
| `balanced` | 79.996% | −0.022%p | 2.0713 | 1.285ms |
| `accuracy` | 80.408% | −0.030%p | 2.2651 | 1.329ms |

FP16 기본 모드는 **79.996%로 80%에 조금 못 미쳤다**. 이번 측정에서 정확도 80% 이상과 낮은 연산량을 함께 본 기본 선택은 FP32 `balanced`다.
지연시간을 더 중시하면 별도 검증된 FP16 `accuracy`를 선택할 수 있다.

## 4. 통합한 구현

### 4.1 전체 KD 가중치

정확도가 가장 좋았던 기존 전체 가중치 KD를 사용한다. Teacher는 공식 EMA의 3H 및 전체 토큰 6H이며 손실은 다음과 같다.

```text
Loss = 0.5 × KL(Student 3H, Teacher 3H)
     + 0.5 × KL(Student Drop 6H, Teacher Full 6H)
T = 1
```

**이번 최종 수치는 새 장시간 재학습 결과가 아니다.** 이미 완료된 KD 체크포인트에 고정 정책과 실행 최적화를 통합해 직접 평가했다.
정리한 학습 코드도 포함했지만 이번 작업에서는 2 update의 실제 역전파·저장 실행 점검만 했다.

### 4.2 게이트와 토큰 유지율 공동 조정

3H에서 끝낼 이미지, 작은 6H를 적용할 이미지, 더 많은 토큰을 줄 이미지에 계산을 배분한다.
새 gate 네트워크를 학습하지 않고 기존 상위 10개 클래스 확률을 재정규화한 entropy를 사용한다.

경계는 `codex/target80-results`의 고정 정책에서 가져왔다. 그때의 KD 가중치와 이 컴퓨터의 KD 가중치는 다르므로
기존 보고의 80.076%를 이번 결과로 복사하지 않았다. 이번 데이터로 경계나 epoch를 다시 선택하지 않고 새로 측정했다.

### 4.3 이미지 임베딩 재사용

원본 실행은 3H와 6H에서 같은 이미지의 patch/position embedding을 다시 계산한다.
통합 구현은 3H에서 만든 **384채널 원본 임베딩**을 보존하고, 6H에서 여기에 3H 특징의 recycling projection을 더한다.
6H에 진입하는 이미지마다 중복 Conv2d **0.057803 GMACs**를 줄이며 토큰 특징의 의미나 위치를 바꾸지 않는다.

### 4.4 고정 폭 모듈과 CUDA Graph

원본의 동적 폭 가중치를 192채널·384채널의 일반 PyTorch 모듈로 각각 변환했다.
추론 중 가중치 인덱싱, attention hook 등록, 토큰 선택용 `mask.nonzero()`를 없애고 고정 길이 Top-K gather를 쓴다.
좁은 경로와 유지율별 넓은 경로는 CUDA Graph로 캡처하고 게이트에 필요한 scalar 읽기와 실제 분기는 남긴다.

**3H/6H는 블록 수가 아니라 attention head 수**다. 두 경로 모두 12개 블록을 사용한다.
패치 순위는 3H 마지막 블록의 head 평균 CLS attention으로 정하고, 선택된 토큰 집합은 6H의 모든 블록에서 유지한다.
Global CLS나 블록별 점진적 제거를 적용한 구조가 아니다.

## 5. 보조 결과와 제외한 실험

이번 조건에서 기존 KD 고정 50%는 **79.670%·2.1014 GMACs**였다.
통합 기본은 이 중간 대조군 대비 **+0.348%p**, 연산량 **1.43% 감소**지만, 주 결론은 1절의 **main 원본 대비**로 판단한다.
과거 KD 기록 79.672%와 이번 조건부 실행 79.670%는 정답 수 1장 차이다.
이번 원본 KD와 통합 `kd50`/KD `full`은 5만 장에서 예측·정오답·분기가 이미지별로 모두 일치했다.

| 기본 경로에서 제외 | 판단 |
| --- | --- |
| Global CLS | 기존 KD 대비 정확도 차이가 매우 작고 실행 시간이 증가 |
| Slow-fast·Global CLS 이어받기 | 완료된 비교에서 기존 KD를 넘지 못하고 실행 시간이 증가 |
| 부분 adapter·LoRA | 학습 파라미터는 줄었지만 전체 KD보다 정확도가 낮고 추론 연산 절감 방법은 아님 |
| CE·Full·Feature 추가 손실 | 주 지표에서 기존 KD를 일관되게 개선하지 못함 |
| Layer-wise Drop | 일부 75% 설정의 작은 이점은 있었지만 공동 정책보다 더 효율적이라는 직접 근거가 없어 통합하지 않음 |

과거 실험·문서·중복 runner는 **이 브랜치에서만** 정리했다. 기존 브랜치와 Git 이력에 근거를 보존했고,
ignored 데이터·체크포인트·실험 원자료는 삭제하지 않았다.

## 6. 측정 조건과 지표 해석

| 항목 | 조건 |
| --- | --- |
| 환경 | RTX 3070 Ti Laptop 8GB, Python 3.12.3, PyTorch 2.5.1+cu124 |
| 정확도 | ImageNet-1K validation 50,000장, seed-42 동일 순서, batch 64 |
| FP32 | AMP off, TF32 허용 |
| 시간 | 같은 1,000장, batch 1, warmup 50장, 순서를 바꾼 3회 반복 평균의 중앙값 |
| 시간 포함 | 3H·gate 동기화·토큰 선택·recycling·필요한 6H·Graph 입력/출력 복사 |
| 시간 제외 | 이미지 읽기·전처리·입력 H2D·최종 argmax·Graph 준비 비용 |
| GPU 공유 검사 | 다른 GPU compute 프로세스를 1초 간격으로 감시; 관측되면 `clean: false` |
| GMACs | 실제 분기·토큰 수를 반영한 이미지당 평균 conv/matmul MACs |

1 GMAC은 10억 번의 multiply-accumulate이며 보통 2 GFLOPs에 대응한다.
이 분석식은 norm·softmax·GELU·정렬/gather·메모리 이동·CPU 호출 비용을 포함하지 않는다.
**GMACs 감소와 실제 가속을 구분**하며 Graph 준비·warmup 시간은 JSON의 `setup_seconds`에 따로 기록한다.

처리량은 batch-1 순차 images/s이며 배치 처리량이 아니다. 메모리는 PyTorch allocated 통계다.
폭별 가중치 복사와 Graph 버퍼 때문에 **메모리는 증가**했다. 입력 1,000장의 GPU 캐시를 제외한 측정 peak는
원본 약 **98.5 MiB**, FP32 Graph 약 **165.6 MiB**, FP16 Graph 약 **112.0 MiB**였다.
Graph 풀이 선할당되므로 incremental peak만 보고 메모리가 절감됐다고 해석하지 않는다.

이 validation은 이전 실험에서 반복 사용했다. 단일 KD seed의 관측이며 새로운 독립 테스트나 작은 정확도 차이의 확증 결과로 주장하지 않는다.
본 브랜치의 통합 모델을 ImageNet-V2에서 새로 평가하지는 않았다. 다른 브랜치의 V2 결과를 이번 모델의 검증으로 대신하지 않는다.

## 7. 설치와 입력 준비

```bash
bash scripts/bootstrap.sh
```

공식 ThinkingViT revision은 `9254c4950d3159937ad682592d58a2be69708c42`로 고정한다.
기존 checkout이 다른 revision이면 스크립트는 그 checkout을 자동으로 바꾸지 않고 종료한다.
공식 EMA는 `checkpoints/thinkingvit_3h6h_ema.pth`, 학습된 KD는 `checkpoints/kd_student.pth`에 둔다.
가중치는 Git에 포함하지 않는다. 이 컴퓨터에는 완료된 KD 가중치를 해당 위치에 복사해 두었다.

ImageNet 사용 승인과 인증을 완료한 뒤 필요한 split을 준비한다.

```bash
# 평가만 할 때
.venv/bin/python scripts/prepare_imagenet.py --splits validation
# KD 학습까지 할 때 추가
.venv/bin/python scripts/prepare_imagenet.py --splits train
```

기본 데이터 위치는 `data/imagenet1k/{train,validation}/<class>/...`다. 기존 데이터가 준비돼 있으면 재다운로드할 필요가 없다.
모델·가중치 해시는 [원본 기준](reports/original-main-protocol.json), [FP32 평가](reports/accuracy-fp32.json)에 기록했다.

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
현재 단위 테스트 12개, 실제 GPU 원본/고정 폭/Graph 출력 검증, 5만 장 이미지별 대조군 일치 검사를 통과했다.

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

| 경로 | 내용 |
| --- | --- |
| [`efficient_vit/policy.py`](efficient_vit/policy.py) | 고정 정책·entropy·MACs 산식 |
| [`efficient_vit/model.py`](efficient_vit/model.py) | 원본 로딩·고정 폭 변환·embedding 재사용 |
| [`efficient_vit/runtime.py`](efficient_vit/runtime.py) | eager/Graph 조건부 추론 |
| [`efficient_vit/verify.py`](efficient_vit/verify.py) | main 원본 경로와 출력 비교 |
| [`efficient_vit/__main__.py`](efficient_vit/__main__.py) | 검증·평가·시간 측정 CLI |
| [`efficient_vit/train.py`](efficient_vit/train.py) | 보존한 전체 KD 학습 recipe |
| [`scripts/report.py`](scripts/report.py) | 공유용 요약 생성 |
| [`reports/RESULTS.md`](reports/RESULTS.md) | 원본 기준 주 비교·KD 보조 비교·p95·처리량·메모리 |
| [`reports/original-main-protocol.json`](reports/original-main-protocol.json) | main revision·체크포인트 해시·원본 설정 |
| [`reports/original_equivalence.json`](reports/original_equivalence.json) | 공식 원본에 실행 최적화만 적용한 5만 장 예측·분기 일치 |
| [`reports/native_equivalence.json`](reports/native_equivalence.json) | KD 원본/통합 대조군의 5만 장 일치 |
| [`reports/implementation_sha256.json`](reports/implementation_sha256.json) | 검증한 구현 해시 |

상세 요약 JSON은 `reports/`, 이미지별 결과·학습 checkpoint·시간 원자료는 ignored `results/efficient-kd/`에 있다.
이전 실험과 선택 근거는 `experiment/drop-ft`, `codex/target80-results`, `experiment/allocation-rule`,
`codex/layerwise-drop-results`, `experiment/h1-diagnosis` 브랜치에 보존돼 있다.
