# Evo-ViT 방법 적용 및 재현

학습 기준은 기존 전체 KD(`drop_ft_experiments.md`의 kd)다. 손실 추가나 부분 동결 없이
공식 EMA에서 각 방법을 독립 학습한다. 실행 상태와 성능은 [자동 결과 문서](evolution_results.md)를 참조한다.

## 적용 범위

`global_cls`는 새 CLS 파라미터를 추가하지 않는다. 기존 3H의 각 블록에서 head 평균 CLS
attention을 추출하고 첫 블록의 attention으로 초기화한다. 이후
`global = 0.5 * previous_global + 0.5 * current_attention`으로 누적한다.
3H 특징 및 분류는 그대로 계산하며, 누적 점수로 6H 입력 Top-K만 바꾼다.
이것은 Evo-ViT Eq.4를 ThinkingViT의 3H→6H 선택에 옮긴 ablation이다.

`evo_layer`/`evo_stage`는 6H에서 선택하지 않은 패치도 원래 공간 슬롯에 유지한다.
비선택 패치의 global attention 정규화 가중합으로 대표 토큰을 만들고,
CLS·선택 패치·대표 토큰만 실제 Transformer 블록에 넣는다.
대표 토큰의 출력에서 입력을 뺀 **잔차**를 각 비선택 패치에 더한다(Eq.5–7).
대표 토큰 자체의 출력으로 패치를 덮어쓰지 않는다. global attention 갱신은 선택 패치에만 적용한다.
재선택 시 기존 placeholder도 다시 선택될 수 있다.

대표 토큰까지 동일 계산 토큰 예산에 포함한다. 유지율 50%이면 기존 Drop은
`CLS+98 patches`, slow-fast는 `CLS+97 patches+1 representative`다.
대표 토큰 생성·gather/scatter·attention 추출에 추가 비용이 있으므로 토큰 예산 일치가
실제 GPU 시간 일치를 뜻하지 않는다. 토큰을 모두 유지하면 대표 토큰 없이 원본 전체 경로를 실행한다.

`evo_schedule`은 논문의 **layer-to-stage training**을 3에폭에 비례 적용한다.
첫 2에폭은 매 블록에서 재선택하고 마지막 1에폭은 4블록마다 재선택한다.
각 블록의 slow-fast 갱신과 attention EMA는 계속 적용한다. 추론은 stage 방식이다.
이는 뒤로 갈수록 토큰 수를 계속 줄이는 pruning 스케줄과 다르다.
동일 학습 가중치에서 layer 방식도 추가 평가하며, 이는 선택 주기의 추론 비교다.
학습 스케줄만의 효과를 분리한 별도 layer-only 학습 대조군은 아니다.

공식 Evo-ViT DeiT 코드는 처음 4블록을 전체 토큰으로 처리하고 placeholder를 그대로 보존하는
변형을 사용한다. 이번 구현은 3H에서 이미 얻은 점수로 6H 시작부터 압축하며,
논문 수식의 대표 토큰 잔차 갱신을 사용한다. 공식 DeiT 모델 재현이 아닌 ThinkingViT 적용 실험이다.
손실·ratio·초기화와 학습량은 기존 KD와 동일하게 유지한다.

## 실행

```bash
.venv/bin/python baseline/run_evolution_study.py
```

실행 순서는 기존 KD 가중치의 선택 방식 교체 평가 → global CLS 3에폭 학습·평가 →
slow-fast/layer-to-stage 3에폭 학습·평가다. 학습 유지율은 batch마다 25/50%에서 선택하며,
학습 후 평가는 25/50/75%를 사용한다. 평가에서는 seed-42 순서의 validation 50,000장,
FP32·TF32 허용, entropy τ=1을 고정한다. 단독 6H 정확도와 routed 정확도를 모두 저장한다.
batch-1 지연은 같은 모델·같은 입력에서 full/각 방법의 실행 순서를 순환하며 측정한다.
50장 warmup 후 1,000장을 사용하고 전처리·H2D는 제외하며 routing 동기화는 포함한다.
학습 전 방식 교체 결과와 새 방법으로 다시 학습한 결과를 구분해야 한다.
실행 중 다른 분류기 학습이 같은 GPU를 사용하는 것이 관측됐다. 동시 작업이 있는 시간 측정은
GPU 경합의 영향을 받으므로 전용 GPU에서의 지연으로 해석하지 않는다. 후속 평가 summary에는
지연 측정 전후의 다른 GPU 프로세스를 기록한다.

실패 시 로그를 확인하고 같은 runner 명령으로 재개할 수 있다. 완료된 단계는 다시 실행하지 않는다.
optimizer·schedule·RNG·batch 위치를 복원하지만 worker augmentation의 bit 단위 재현은 보장하지 않는다.
학습과 평가는 모두 장시간 작업이며 완료 여부는 실제 `complete.json`/`summary.json`으로 판정한다.

## 구현 검증

- 전체 단위 테스트 40개 통과.
- 실제 GPU: 3H 특징 불변, 기존 Drop 출력 일치, 유지율 100%에서 전체 경로 출력 일치.
- 새 네 method의 BF16 역전파에서 유한 gradient 확인.
- global CLS 및 evo_schedule 각각 실제 train 이미지로 KD optimizer 2 update 및 체크포인트 저장 확인.
- 원자료의 `implementation_verification.json`에 검증 기록과 코드 SHA-256을 보존한다.

참고: [논문](https://arxiv.org/abs/2108.01390),
[공식 코드](https://github.com/YifanXu74/Evo-ViT/blob/main/deit/evo_deit.py).
