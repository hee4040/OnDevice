# 통합 범위와 비교 기준

새 브랜치 `integration/efficient-kd`는 다음 네 가지를 통합한다.

1. 전체 가중치 KD: 0.5 KL(3H)+0.5 KL(Drop 6H), 공식 EMA Teacher, T=1.
2. 종료 게이트와 토큰 유지율 공동 조정: entropy가 중간이면 작은 6H, 높으면 큰 6H.
3. 고정 폭 Linear/LayerNorm/QKV와 고정 길이 Top-K gather, CUDA Graph 실행.
4. 3H에서 만든 원본 이미지 embedding을 6H에 재사용해 중복 Conv2d를 제거한다.

기존 KD 가중치를 사용한다. `efficient_vit.train`은 원래의 KD 학습만 재현하며 정책은 학습 후 적용한다.
Global CLS·slow-fast·부분 adapter·CE/feature 추가 손실은 기본 구현에서 제거했다.
Layer-wise 75%는 동일 비용 uniform 대비 작은 개선이 있었으나 공동 정책보다 효율적이라는 직접 증거가
없어 기본 경로에 섞지 않았다. 조합 후 성능은 이 브랜치에서 직접 평가한다.

## 고정한 정책

| 이름 | 종료 | 6H 유지율 | 용도 |
| --- | --- | --- | --- |
| economy | entropy < 1 | 25% | 연산량 절감 우선 |
| balanced | entropy < 0.8 | 1.25 미만 25%, 이상 50% | 기본 공동 정책 |
| accuracy | entropy < 0.75 | 1.15 미만 25%, 1.75 미만 50%, 이상 75% | 정확도 우선 |
| kd50 | entropy < 1 | 50% | 기존 KD 대조군 |
| full | entropy < 1 | 100% | 같은 KD 가중치 전체 토큰 대조군 |

경계는 `origin/codex/target80-results`의 고정 정책에서 가져왔다. 그때 가중치 SHA-256은
`a14958ad…6562bd`였고 이번 로컬 KD는 `37e9a36a…d628ae`다. 기존 80.076%를 복사해서 이번 성능으로
표시하지 않는다. 새 측정만 `reports/RESULTS.md`에 기록한다. 이번 validation으로 경계/epoch를 다시
선택하지 않는다. 이 데이터는 이전 실험에서 반복 사용했으므로 독립적인 새 평가라고 주장하지 않는다.

## 실행 구조

폭 192/384 가중치를 각각 일반 PyTorch 모듈로 변환한다. 좁은 Q/K/V는 원본 세 projection의
각 시작점에서 192개씩 골라야 한다. 추론 중 가중치 인덱싱·hook·토큰 mask.nonzero를 없앴다.
동일한 폭별 가중치를 중복 보관하므로 GPU 상주 메모리는 원본 단일 모델보다 늘 수 있다.
학습 파라미터 수를 줄인 방법은 아니다.

batch-1은 좁은 경로와 유지율별 넓은 경로를 CUDA Graph로 각각 캡처한다. 게이트 scalar 읽기 1회·
분기·출력 복사는 실제 시간에 포함한다. Graph는 배치 1 전용이며 배치 평가는 eager를 사용한다.
검증한 고정 폭 eager와 직접 CUDA Graph만 제공한다. 캡처 준비 시간은 steady-state 시간에 합산하지 않는다.
원본 대비 특징·logits·attention·entropy 및 25/50/75/100% 출력을 검증한다. benchmark 전 실제 이미지
32장에서도 eager와 가속 경로의 logits·선택을 비교한다. FP16은 별도 검증·평가한다.
출력은 다음 Graph 호출 뒤에도 유효하도록 복사한다. embedding은 384채널로 한 번 계산한 원본을 보존한다.
6H에서는 그 원본에 recycling projection을 더한다. 재학습이나 토큰 특징 변경 없이 같은 embedding
Conv2d를 한 번 덜 수행하므로, 6H에 진입하는 이미지마다 0.057803 GMACs를 줄인다.

## 지표

- 정확도: validation 50,000장, seed-42의 같은 순서.
- GMACs: 실제 분기·토큰 수·한 번 계산한 384채널 embedding을 포함한 conv/matmul MACs. native 대조군은 embedding 재계산을 포함한다. norm·softmax·sort/gather·
  메모리 이동·CPU 호출 비용 제외. 1 MAC은 보통 2 FLOPs다.
- 시간: 같은 1,000장·batch 1·warmup 50장·순서를 바꾼 3회 평균의 중앙값.
  읽기/전처리/H2D/최종 argmax 제외, 3H·게이트·선택·recycling·6H 포함.
- 처리량: `serial_images_per_s`는 batch-1 순차 처리량이며 배치 처리량이 아니다.
- 메모리: 상주 할당과 peak를 따로 기록한다. Graph 선할당이 있으므로 incremental peak만 비교하지 않는다.
  다른 프로세스 메모리는 PyTorch 통계에 포함되지 않는다.

## 보존한 근거

- KD 및 제거한 실험: `experiment/drop-ft` / `d9e28b6`.
- 공동 정책: `codex/target80-results` / `f2df226`.
- ImageNet-V2 확인: `experiment/allocation-rule` / `9e7e2c0`.
- 레이어별 제거: `codex/layerwise-drop-results` / `424b185`.
- 실행 최적화: `experiment/h1-diagnosis` / `9d8f655`.

기존 브랜치·Git 이력과 ignored 데이터/가중치/원자료는 유지했다. 새 브랜치에서만 오래된 문서·그림·
실험 runner·중복 분석을 제거했다. 학습 재개는 optimizer/schedule/RNG/batch 위치를 복원하지만
worker augmentation의 bit 단위 재현은 보장하지 않는다.
