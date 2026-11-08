# 80% 목표 실험 실행 안내

기존 파인튜닝 학생의 종료 기준과 token drop 유지율을 함께 조정합니다. 측정한 규칙은 entropy < 0.8에서 종료, 0.8~1.25에서 패치 49개 유지, 1.25 이상에서 패치 98개 유지입니다. 가중치·crop 0.9·bicubic은 유지합니다.

## 1. 환경 준비

저장소 루트에서 실행합니다. Python 3.10, PyTorch 2.5.1+cu124, numpy 2.2.6, matplotlib 3.10.7 환경으로 측정했습니다. 설치 목록은 `requirements.txt`에 고정되어 있습니다. PyPI의 다른 timm 대신 공식 ThinkingViT 저장소 안의 수정된 timm을 사용합니다.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 2. GPU 없이 수치 재계산

```bash
.venv/bin/python scripts/restore_target80_artifacts.py
.venv/bin/python baseline/target80_gate_search.py \
  --source results/20261107T091527.251697Z-adaptive-drop \
  --output results/target80-replay/gate_search
```

`baseline/reports/2026-11-08/target80/replay/`의 압축 파일은 이미지·가중치가 아닌 실험 수치 자료입니다. 모든 항목의 SHA-256을 검증한 뒤 복원합니다. 원래 캐시 해시와 정책 선택 결과를 그대로 확인할 수 있습니다. 기존 `results/`에 해당 이름의 실험이 있으면 복원이 중단됩니다. 이때 `--destination-root /다른/빈/폴더`를 사용하고 아래 입력 경로를 그 위치에 맞춥니다.

복원된 `gate_search/summary.json`의 서버 경로 하나만 현재 위치로 조정합니다. 원래 값과 변경 내역은 `restore_record.json`에 남고, 압축 파일의 원본은 바꾸지 않습니다.

저장된 실제 GPU 예측·시간 원자료로 표와 그림을 다시 만들려면:

```bash
.venv/bin/python baseline/report_target80.py --experiment results/20261108-target80
```

이 명령은 공유용 보고서 파일도 다시 생성합니다. 사후 교차검증을 새로 수행하려면 원래 `diagnostics/`를 덮어쓰지 않도록 별도 실험 복사본을 사용합니다.

## 3. 실제 GPU 재실행

다음 입력을 준비합니다.

- `third_party/ThinkingViT`: 공식 코드. 기존 저장소의 `scripts/bootstrap.sh`와 베이스라인 안내를 참고합니다.
- `checkpoints/thinkingvit_3h6h_ema.pth`: 공식 EMA 가중치.
- `student.pth`: 측정에 쓴 학생. SHA-256 `a14958ad4da105c9e0290aacadd6c7d83ce6003831090351cfc0fd13eb6562bd`.
- ImageNet-1K validation 50,000장: 저장된 ImageFolder class/index 및 파일명이 일치해야 합니다.

학생 가중치는 이 브랜치에 배포하지 않습니다. 다른 가중치로 얻은 결과를 이 수치의 재현으로 취급하지 않습니다. 새 가중치나 수치적으로 다른 캐시를 연구할 때는 별도 실험 설정과 결과 폴더를 만들고 변경을 기록해야 합니다.

2절의 수치 자료 복원과 `results/target80-replay/gate_search` 생성을 마친 후 실행합니다. 아래 두 경로는 사용 환경에 맞춥니다.

```bash
export ONDEVICE_DATA_DIR=/path/to/imagenet1k
export ONDEVICE_STUDENT=/path/to/student.pth

CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate_target80.py \
  --data-dir "$ONDEVICE_DATA_DIR" \
  --repo third_party/ThinkingViT \
  --checkpoint checkpoints/thinkingvit_3h6h_ema.pth \
  --weights "$ONDEVICE_STUDENT" \
  --source results/20261107T091527.251697Z-adaptive-drop \
  --experiment results/target80-replay

CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/measure_target80.py \
  --data-dir "$ONDEVICE_DATA_DIR" \
  --repo third_party/ThinkingViT \
  --checkpoint checkpoints/thinkingvit_3h6h_ema.pth \
  --weights "$ONDEVICE_STUDENT" \
  --source results/20261107T091527.251697Z-adaptive-drop \
  --experiment results/target80-replay

.venv/bin/python baseline/diagnose_target80.py \
  --source results/20261107T091527.251697Z-adaptive-drop \
  --experiment results/target80-replay
```

전체 학습은 하지 않고 기존 학생을 평가합니다. 실행기에서는 이미지마다 선택된 6H 경로 하나만 수행합니다. 평가 중에는 고정한 정책을 다시 조정하지 않습니다.

## 4. 해석과 재현 기록

- `gpu_accuracy/comparison.csv`: 실제 GPU 결과. 주 정책은 40,038 / 50,000 = 80.076%입니다.
- `gate_search/`: 보정용에서 고른 정책과 캐시 기반 비교. 실제 그룹 배치 실행의 미세한 수치 차이는 GPU 보고서에 별도로 남겼습니다.
- `diagnostics/`: 구간별 개선·손상과 사후 crossfit 결과. 서로 다른 시드도 같은 50,000장을 다시 나눈 것입니다.
- `latency.json`: 네 정책 × 세 반복 × 동일 1,000장의 시간 원자료. 이미지 전처리·입력 H2D는 제외합니다.
- `manifest.json`: 측정 당시 코드·가중치 해시. 공유용 README·복원 도구는 이후 추가된 포장 작업입니다.

기존 validation을 반복 탐색한 결과이므로 새 독립 평가로 주장하지 않습니다. 분석 MAC 예산과 실제 지연시간도 구분합니다. 주 정책의 평균 MAC은 기준보다 낮았지만 실제 시간은 8.8% 증가했습니다.

## 5. 추가 학습 준비 상태

`finetune_target80.py`는 50% 유지율 전용 KD와 CE+KD 두 변형입니다. 각각 10단계에서 손실·gradient·체크포인트 저장을 점검했습니다. 전체 1 epoch 학습은 수행하지 않았고, 보고된 80% 결과에도 이 점검용 가중치를 사용하지 않았습니다. 전체 추가 학습의 효과는 미검증입니다.
