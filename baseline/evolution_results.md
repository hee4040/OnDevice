# Global CLS 및 Evo-ViT 실험

갱신(UTC): 2026-11-15T06:36:24.695992+00:00

상태: 모든 학습·평가 완료

기준은 기존 전체 KD: Drop 50%·τ=1 Top-1 79.672%.
공식 EMA에서 각각 새로 시작해 전체 backbone을 학습한다. ImageNet train 1,281,167장, 3에폭,
batch 128 / micro 32, AdamW lr=5e-5, weight decay=0, warmup 500, cosine, seed 42, BF16.
손실은 0.5 KL(3H)+0.5 KL(압축 6H), T=1. 유지율은 batch별 25/50%로 기존과 같다.

- global_cls: 3H의 12블록 CLS attention을 decay=0.5 EMA로 누적하고 고정 Top-K로 6H를 실행한다.
- evo_schedule: 같은 global CLS로 시작하고 6H에서 대표 토큰의 잔차를 비선택 토큰에 전파한다.
  첫 2에폭은 블록별 재선택, 마지막 1에폭은 4블록마다 재선택한다. 추론은 4블록 단위다.
- evo_layer 평가는 같은 evo_schedule 가중치에서 블록별 재선택을 적용하는 추론 비교다.
  별도 학습된 layer-only 대조군이 아니므로 학습 스케줄 효과만을 분리하지 않는다.

3H의 토큰 수는 유지한다. 압축은 재활용 6H 입력을 만든 뒤 시작한다. 대표 토큰 1개도
기존 Drop과 같은 예산에 포함하므로 50%는 CLS 1+선택 패치 97+대표 1이다.
논문 Eq.4–7 및 layer-to-stage(200/300 epoch)를 3에폭에 비례 적용한 ThinkingViT 변형이다.
공식 DeiT 구현은 앞 4블록을 dense로 두고 placeholder를 그대로 보존하는 변형을 사용한다.
이번 코드는 3H 정보를 활용해 6H 첫 블록부터 압축하며 논문의 대표 토큰 잔차를 갱신한다.
논문의 전체 DeiT 학습 결과 재현으로 해석하지 않는다.

평가: seed-42 validation 50,000장, FP32·TF32 허용, τ=1 고정. 임계값·epoch 선택 없음.
기존 validation을 재사용하는 단일 seed 탐색이며 새 미사용 테스트 성능으로 보지 않는다.

| 평가 | 표본 수 | 경로 | 전체 6H 경로 Top-1 | τ=1 Top-1 |
| --- | ---: | --- | ---: | ---: |
| evo_schedule | 50000 | narrow_correct | 73.496% | 73.496% |
| evo_schedule | 50000 | wide_correct | 80.972% | 79.902% |
| evo_schedule | 50000 | evo_layer_25_correct | 78.872% | 78.216% |
| evo_schedule | 50000 | evo_layer_50_correct | 80.630% | 79.634% |
| evo_schedule | 50000 | evo_layer_75_correct | 80.992% | 79.894% |
| evo_schedule | 50000 | evo_stage_25_correct | 78.816% | 78.194% |
| evo_schedule | 50000 | evo_stage_50_correct | 80.552% | 79.572% |
| evo_schedule | 50000 | evo_stage_75_correct | 81.012% | 79.898% |

evo_schedule batch-1 지연(ms), 같은 모델의 τ=1·50% 유지·전처리/H2D 제외:
```json
{
  "samples": 1000,
  "warmup": 50,
  "batch_size": 1,
  "ratio": 0.5,
  "threshold": 1,
  "preprocessing_h2d_excluded": true,
  "mean_ms": {
    "full": 7.042615755693987,
    "evo_layer": 10.336239429656416,
    "evo_stage": 9.719791045645252
  },
  "other_gpu_processes_before": [],
  "other_gpu_processes_after": []
}
```

| existing_kd | 50000 | narrow_correct | 73.498% | 73.498% |
| existing_kd | 50000 | wide_correct | 81.092% | 80.040% |
| existing_kd | 50000 | last_cls_50_correct | 80.688% | 79.672% |
| existing_kd | 50000 | global_cls_50_correct | 80.652% | 79.642% |
| existing_kd | 50000 | evo_layer_50_correct | 79.248% | 78.596% |
| existing_kd | 50000 | evo_stage_50_correct | 79.404% | 78.682% |

existing_kd batch-1 지연(ms), 같은 모델의 τ=1·50% 유지·전처리/H2D 제외:
```json
{
  "samples": 1000,
  "warmup": 50,
  "batch_size": 1,
  "ratio": 0.5,
  "threshold": 1,
  "preprocessing_h2d_excluded": true,
  "mean_ms": {
    "full": 46.63197077717632,
    "last_cls": 47.31289431452751,
    "global_cls": 49.54010211979039,
    "evo_layer": 57.02442050189711,
    "evo_stage": 53.491601679241285
  }
}
```

| global_cls | 50000 | narrow_correct | 73.526% | 73.526% |
| global_cls | 50000 | wide_correct | 81.104% | 79.988% |
| global_cls | 50000 | last_cls_25_correct | 79.006% | 78.280% |
| global_cls | 50000 | last_cls_50_correct | 80.586% | 79.564% |
| global_cls | 50000 | last_cls_75_correct | 81.066% | 79.934% |
| global_cls | 50000 | global_cls_25_correct | 79.216% | 78.448% |
| global_cls | 50000 | global_cls_50_correct | 80.744% | 79.678% |
| global_cls | 50000 | global_cls_75_correct | 81.096% | 79.984% |

global_cls batch-1 지연(ms), 같은 모델의 τ=1·50% 유지·전처리/H2D 제외:
```json
{
  "samples": 1000,
  "warmup": 50,
  "batch_size": 1,
  "ratio": 0.5,
  "threshold": 1,
  "preprocessing_h2d_excluded": true,
  "mean_ms": {
    "full": 7.23660413431935,
    "last_cls": 7.523369460366666,
    "global_cls": 8.765000846702605
  },
  "other_gpu_processes_before": [],
  "other_gpu_processes_after": []
}
```


## 실행 기록

| 단계 | 시작 UTC | 종료 UTC | 상태 |
| --- | --- | --- | --- |
| existing_kd: validation | 2026-11-14T21:06:23.510478+00:00 | 2026-11-14T21:21:23.552116+00:00 | complete |
| global_cls: KD train | 2026-11-14T21:21:23.553150+00:00 | 2026-11-15T01:44:24.072529+00:00 | complete |
| global_cls: validation | 2026-11-15T01:44:24.073325+00:00 | 2026-11-15T01:49:54.083268+00:00 | complete |
| evo_schedule: KD train | 2026-11-15T01:49:54.084058+00:00 | 2026-11-15T06:30:24.662719+00:00 | complete |
| evo_schedule: validation | 2026-11-15T06:30:24.667844+00:00 | 2026-11-15T06:36:24.685613+00:00 | complete |

원자료: `results/evo-vit-20261115`. 각 단계 명령·종료 코드: `study.json`.
학습 진행: 각 `train/train_log.json`, 완료: `train/complete.json`. 체크포인트에 optimizer·RNG 포함.
중단 재개 시 augmentation의 bit 단위 재현은 보장하지 않는다. Commit·push 없음.

참고: [Evo-ViT 논문](https://arxiv.org/abs/2108.01390),
[공식 구현](https://github.com/YifanXu74/Evo-ViT/blob/main/deit/evo_deit.py).
