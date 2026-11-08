# 결과 파일 안내

날짜별 폴더에 실험 결과 요약이 들어 있습니다. 이미지별 원자료와 가중치는 용량 때문에 올리지 않았고, 실행한 PC의 `results/`에 있습니다.

| 폴더 | 실험 | 설명 문서 | 주요 파일 |
| --- | --- | --- | --- |
| `2026-10-14/` | 베이스라인 정확도·지연 (RTX 3090, 준철) | [`docs/baseline.md`](../../docs/baseline.md) | `accuracy.json` 정확도, `latency.json` 지연, `provenance.json` 환경·해시 |
| `2026-10-15/` | 실패 분석 (준철) | [`failure_analysis.md`](../failure_analysis.md) | `failure_analysis.json`, `stage_analysis.json`, 사례 CSV |
| `2026-10-27/` | CLS attention 분석, 패치 가림 실험 (준철) | main 브랜치 커밋 `ac7ff72` | `attention_maps.png`, `occlusion_summary.json` |
| `2026-11-01/` | RTX 4090 재현, H1 76~100% sweep (장준용) | [`rtx4090_reproduction.md`](../rtx4090_reproduction.md) | `h1_ratio_sweep.csv`, `h1_ratio_sweep_76_100.png` |
| `2026-11-02/` | H1 진단 실험 1~4 (장준용) | [`h1_diagnosis.md`](../h1_diagnosis.md) | `h1_diagnosis.csv` 설정별 정확도, 그림 4장, `h1_diagnosis_key_numbers.json` |
| `2026-11-03/` | 후속 실험 5~7: 추론 시간, 부풀림 확인, 파인튜닝 (장준용) | [`h1_diagnosis.md`](../h1_diagnosis.md) | 아래 표 |
| `2026-11-08/` | 실험 2 GPU 최적화: torch.compile, CUDA Graphs, FP16 (Kaggle T4, 장준용) | [`gpu_optimization.md`](../gpu_optimization.md) | 아래 표 |

## `2026-11-03/` 파일

| 파일 | 내용 |
| --- | --- |
| `followup_key_numbers.json` | 같은 연산량 비교, 부트스트랩 구간, 반 나눠 검증, 파인튜닝 전후 수치 전부 |
| `followup_pareto.png` | 연산량-정확도 그림 (τ 조정 곡선, 토큰 제거 학습 전·후) |
| `followup_latency.png` | 6H 단계 시간 비교 그림 (이론값, GPU, CPU 4코어, CPU 1코어) |
| `latency.json` | 6H 단계별 시간 원자료 |
| `pipeline_latency.json` | 방법별 이미지 한 장 전체 추론 시간 원자료 (검증 1,000장) |
| `official_latency.csv` | 팀 기준 `evaluate.py --mode latency` 결과 (RTX 4090) |
| `drop_eval_official.json`, `drop_eval_finetuned.json` | 원래·파인튜닝 모델의 3H, 6H, 토큰 제거 정확도 (파인튜닝 가중치 SHA-256 포함) |
| `finetune_log.json` | 파인튜닝 손실 기록 (100단계마다) |

## `2026-11-08/` 파일

이미지별 원자료(`per_image.pt`)는 Kaggle 노트북 출력에 있습니다.

| 파일 | 내용 |
| --- | --- |
| `exp2_key_numbers.json` | 실행 방식별 단계 시간, 사진 한 장 평균 시간, 같은 시간 비교와 부트스트랩 구간, FP16 정확도와 답이 바뀐 사진 수 |
| `exp2_time.png` | 실행 방식별 사진 한 장 평균 시간 그림 (τ = 1.0) |
| `exp2_batch.png` | 한 번에 처리하는 사진 수(1, 8, 32)에 따른 토큰 제거의 6H 시간 그림 |
| `exp2_latency.json` | 단계별 시간 원자료 (`optimize_latency.py`, 녹화·컴파일 실패 기록 포함) |
| `drop_eval_official_t4.json`, `drop_eval_finetuned_t4.json` | T4 FP32 정확도 (원래·파인튜닝 모델) |
| `drop_eval_official_fp16.json`, `drop_eval_finetuned_fp16.json` | T4 FP16 정확도 (원래·파인튜닝 모델) |
