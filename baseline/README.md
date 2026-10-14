# ThinkingViT 실제 early-exit baseline

공식 구현과 가중치를 그대로 사용하면서, 3H 예측의 Entropy가 threshold보다 낮으면 6H를 실행하지 않는다. 별도의 학습이나 게이트 모델은 없다. 공식 `model.entropy()`는 상위 10개 클래스 확률을 재정규화한 Entropy를 계산하며, 이 규칙도 그대로 사용한다.

`third_party/ThinkingViT/`와 기존 `scripts/baseline.py`는 수정하지 않는다. `early_exit.py`는 실행 분기만 담당하고 `evaluate.py`는 검증과 측정을 담당한다. eval 모드, 3H → 6H를 지원한다. 레이턴시는 batch-1 경로, 정확도는 배치 내부에서 어려운 이미지만 골라 6H를 실행하는 경로를 사용한다.

저장소 루트에서 실행한다.

```bash
# 같은 validation 이미지 1,000장으로 공식 forward와 실제 early exit 비교
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py --mode latency

# 공식 출력과의 일치 여부만 검증: 기본 1,000장 × threshold 3개
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py --mode verify

# 실제 early exit의 전체 validation 50,000장 정확도 측정
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py --mode accuracy --batch-size 128

# 전체 50,000장에 대해 동일한 배치의 공식 예측도 함께 비교 (추가 추론 필요)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py --mode accuracy --batch-size 128 --compare-official

# 짧은 검증 및 속도 확인
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/evaluate.py --mode latency --samples 100 --warmup 10
```

기본 threshold는 `10.0 0.0 1.0`이며 기존 baseline과 같은 체크포인트·전처리·FP32 설정을 사용한다. latency와 verify의 이미지 순서는 seed 42로 고정하고 사용한 인덱스를 저장한다. accuracy는 전체 50,000장을 사용한다.

latency/accuracy 실행 전에는 threshold마다 기본 32장(측정 이미지 수가 더 적으면 해당 개수)에서 공식 경로와 logits의 근사 일치, Top-1·선택 단계의 정확한 일치, 실제 단계 호출 횟수를 검사한다. 실패하면 측정을 중단한다. batch-1 logits 허용 오차는 `rtol=1e-5, atol=1e-6`이다. accuracy/verify에서는 배치 경로의 출력과 실제 6H 입력 개수도 추가 검사한다. 배치 경로는 6H 입력 크기가 달라지면서 수치 오차가 생길 수 있어 `rtol=1e-4, atol=1e-5`를 사용하며, Top-1·선택 단계는 정확히 일치해야 한다. 검사 수는 `--verify-samples`로 조정한다. 부분집합 검증은 전체 validation 정확도 검증을 대체하지 않는다.

측정에는 Entropy 판단과 GPU → CPU 동기화 비용을 포함한다. 이미지 전처리와 GPU 전송은 제외하며, 메모리는 모델을 포함한다. 두 분기를 warmup한 뒤 공식 경로와 early-exit 경로를 순서대로 측정하므로 다른 GPU 작업이나 온도·클럭 변화의 영향을 받을 수 있다.

결과는 `results/<시각>-early-exit-<모드>/`에 저장한다. `verification.json`은 출력 일치와 단계 호출 검증 결과, `summary.json/csv`는 측정 결과, `*.timings.json`은 이미지별 시간과 선택 단계다. `environment.json`에는 환경·공식 커밋·체크포인트 및 baseline 코드 해시를 기록한다. 진행 상황은 100장마다, accuracy는 10배치마다 출력한다.

- `stage_2_fraction`: 6H 출력이 선택된 비율. 공식 경로와 early exit에 공통으로 사용한다.
- `stage_2_execution_fraction`: 실제 6H 실행 비율. 공식 경로는 항상 1.0, early exit는 선택 비율과 같다.
- `peak_allocated_mib` / `peak_reserved_mib`: 해당 경로 전체 실행 중 최대값이다. 일부 이미지라도 6H로 진입하면 early exit도 공식 경로와 비슷한 peak를 보일 수 있다.

정확도 기본 batch size는 128, 데이터 로더 worker는 4다. `--batch-size`, `--workers`로 변경할 수 있다. 레이턴시는 이 값과 무관하게 항상 batch-1이며, 단일 GPU에서 측정한다.

배치 크기나 6H로 보내는 이미지 수가 달라지면 GPU 부동소수점 연산의 작은 차이로 경계에 있는 예측이 바뀔 수 있다. `--compare-official`은 정확도 평가의 모든 입력을 같은 배치의 공식 forward에도 전달하고, 공식 Top-1/Top-5·예측 불일치 개수·선택 단계 불일치 개수·최대 logits 차이를 기록한다.
