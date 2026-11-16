# Efficient KD ThinkingViT

기존 전체 KD에 **게이트·토큰 유지율 공동 조정**, **중복 embedding 재사용**, **고정 폭 CUDA Graph 실행**을 통합했다.
정확도·평균 GMACs·실제 시간을 함께 비교하며 절약/균형/정확도 모드를 제공한다.
[이번 브랜치 실측](reports/RESULTS.md) · [방법·범위·평가 한계](docs/method.md)

## 준비

```bash
bash scripts/bootstrap.sh
# ImageNet 접근 승인·인증 후 데이터 준비
.venv/bin/python scripts/prepare_imagenet.py --help
```

공식 ThinkingViT revision은 `9254c4950d3159937ad682592d58a2be69708c42`다.
공식 EMA `checkpoints/thinkingvit_3h6h_ema.pth`와 KD `checkpoints/kd_student.pth`를 준비한다.
가중치는 Git에 포함하지 않는다. 이 컴퓨터에는 기존 완료 KD를 해당 위치에 복사해 두었다.
다른 환경에서는 같은 가중치를 제공하거나 아래 명령으로 학습한다.

```bash
.venv/bin/python -m efficient_vit.train --output results/kd-train
# 학습 후 --weights results/kd-train/student.pth 사용
```

Teacher는 공식 EMA다. 3에폭·batch 128/micro 32·BF16·AdamW 5e-5·weight decay 0·warmup 500·
cosine·seed 42, batch별 25/50% 유지, 0.5 KL(3H)+0.5 KL(Drop 6H)를 사용한다.
`--resume .../resume.pth`로 재개한다. worker augmentation의 bit 단위 재현은 보장하지 않는다.
이번 통합 작업에서는 새 장시간 학습을 시작하지 않고 검증된 기존 KD 가중치를 사용했다.

## 검증·평가·시간 측정

```bash
# 원본 / 고정 폭 / Graph 출력 일치
.venv/bin/python -m efficient_vit validate
.venv/bin/python -m efficient_vit validate --real-images

# 전체 5만 장 정확도·분기별 GMACs
.venv/bin/python -m efficient_vit evaluate --output results/accuracy.json

# 실제 batch-1 시간: 다른 GPU compute 작업이 없을 때
.venv/bin/python -m efficient_vit benchmark --runtime graph --output results/graph.json
.venv/bin/python -m efficient_vit benchmark --runtime eager --output results/eager.json
.venv/bin/python -m efficient_vit benchmark --runtime native --policies kd50 --output results/native.json

# FP16은 별도 검증·평가
.venv/bin/python -m efficient_vit validate --precision fp16
.venv/bin/python -m efficient_vit evaluate --precision fp16 --output results/accuracy-fp16.json

.venv/bin/python -m unittest discover -s tests
```

`--samples 128`은 실행 점검용이며 전체 정확도로 보고하지 않는다.
benchmark는 다른 GPU compute 프로세스를 감시하고 관측 시 `clean: false`로 표시한다.
Graph 캡처 등 준비 비용은 별도다. CPU 디스패치 최적화와 GMACs 절감은 서로 다른 효과다.
Graph는 batch 1 전용이며 정확도 배치 평가는 eager로 실행한다.

## Python

```python
from efficient_vit import load_native, specialize, Engine

native = load_native("checkpoints/thinkingvit_3h6h_ema.pth", "checkpoints/kd_student.pth")
engine = Engine(specialize(native), runtime="graph")
# images: 공식 전처리된 CUDA FP32 [1, 3, 224, 224]
logits, path, entropy = engine.predict(images, policy="balanced")
```

`economy`는 연산 절감, `balanced`는 공동 조정, `accuracy`는 추가 연산을 허용한다.
`kd50`과 `full`은 같은 가중치 대조군이다. Global CLS·slow-fast·adapter·과거 runner는 새 브랜치에서
제거했고 `experiment/drop-ft`에 보존했다. 데이터·체크포인트·실험 원자료는 삭제하지 않았다.
