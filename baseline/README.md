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

## 선택적 토큰 갱신에 앞선 이미지별 분석

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python baseline/analyze_stages.py --batch-size 128
```

전체 validation 50,000장에서 3H와 전체 6H의 예측을 기록한다. 이 오프라인 분석은 모든 이미지에 두 단계를 실행한다. `routed_to_6h`는 기존 Entropy 게이트가 6H를 선택하는지를 뜻하며, 분석 실행 자체가 조기 종료했다는 뜻은 아니다. 기본 threshold는 1.0, 이미지 순서는 기존 batch 128 평가와 같은 seed 42다.

결과는 `results/<시각>-stage-analysis/`에 저장한다.

- `images.csv`: 이미지 경로·정답·3H/6H 예측·정답 여부·Top-5 정답 여부·3H Entropy·게이트 선택·최종 예측·전이 그룹.
- `narrow_cls_attention.npy`: float32 배열 `(50000, 197)`. 3H 마지막 블록 CLS Attention의 head 평균이다. CSV의 `attention_row`와 배열 행이 대응한다. 열 0은 CLS 자체, 열 1~196은 전처리된 이미지의 14×14 패치이며 행 우선 순서다. 패치 열만 남긴 뒤 재정규화하지 않았다.
- `summary.json`: 전체, 6H 진입 대상, 3H 종료 대상을 분리한 네 그룹의 개수와 정확도. 네 그룹은 `both_correct`(정답 유지), `corrected`(오답→정답), `regressed`(정답→오답), `both_wrong`(둘 다 오답)이다.
- `environment.json`, `sample_indices.json`: 환경·공식 커밋·가중치/코드 해시·전처리와 이미지 인덱스.

공식 forward의 두 분류 출력을 hook으로 관찰하며, 마지막 3H 블록의 Q/K로 CLS 행만 추가 계산한다. fused Attention 실행은 유지한다. 첫 배치에서 관찰 전후 출력의 정확한 일치를 검사하고, 모든 배치에서 게이트·출력 선택과 Attention 확률 합을 검사한다. 이 실행시간은 latency 결과로 사용하지 않는다.

완료 전에는 CSV와 Attention 파일에 `partial` 표시를 붙이며, 모든 표본을 처리한 뒤에만 최종 파일명과 `summary.json`을 생성한다. `--samples 128`로 짧게 확인할 수 있지만 전체 validation 분석으로 보고하지 않는다.

2026-10-15 전체 분석 결과는 다음과 같다. batch 128, threshold 1.0, seed 42를 사용했다.

| 3H → 전체 6H | 전체 50,000장 | 6H 진입 대상 18,004장 | 3H 종료 대상 31,996장 |
| --- | ---: | ---: | ---: |
| 정답 → 정답 | 35,522 | 6,965 | 28,557 |
| 오답 → 정답 | 5,198 | 4,330 | 868 |
| 정답 → 오답 | 1,245 | 945 | 300 |
| 오답 → 오답 | 8,035 | 5,764 | 2,271 |

3H Top-1은 73.534%, 전체 6H는 81.440%, 기존 게이트를 적용한 최종 정확도는 80.304%로 이전 baseline과 일치했다. 실제 6H 진입 대상의 순정답 증가는 4,330−945=3,385장이며, 전체 기준 6.770%p 개선이다. 3H 종료 대상의 두 단계 비교는 실제 배포 경로에서는 실행되지 않는 6H를 추가로 계산한 사후 분석이다.

후속 실험의 Correction Retention은 6H 진입 대상 중 교정된 4,330장을 기준으로 계산하고, 기존 6H가 맞힌 11,295장(6,965+4,330)에서 새로 발생하는 오답도 함께 집계한다. 현재 분석만으로 어떤 토큰의 갱신이 교정에 기여했는지는 판단할 수 없다.

[집계 JSON](reports/2026-10-15/stage_analysis.json). 이미지별 CSV와 CLS Attention 원본은 로컬 `results/20261015T041255.578722Z-stage-analysis/`에 있다. Attention 배열은 약 39.4 MB이며, 50,000개 고유 이미지·클래스별 50장·CSV와 Attention의 행 대응·확률 합을 검증했다.

## 실패 분석 보고서

[ThinkingViT 실패 분석](failure_analysis.md)에 전체 오류 통계, 3H 정답→6H 오답 945장의 확률 변화, ImageNet ReaL 재평가, 실제 입력 이미지 8개의 사례를 정리했다. 표본 선택 방법과 판정이 어려운 경우의 한계, 근거 JSON·CSV 및 재현 명령도 보고서에 포함했다.
