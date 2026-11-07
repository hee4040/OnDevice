# Layer-wise Drop: H1 진단 이후 추가 실험

기준 commit: `experiment/h1-diagnosis` / `5d53816`.
작업 브랜치: `codex/layer-wise-drop`. 기존 브랜치의 결과와 코드는 유지한다.
아직 새 정확도·속도 측정치는 없으며, 실행 후 생성되는 결과만 실험 결과로 사용한다.

## 질문과 통제 조건

“모든 Block에서 같은 토큰을 제거해야 할까?” 중 이번에는 **깊이에 따른 유지율**을 검증한다.
토큰의 중요도 순위는 기존과 동일한 3H 마지막 Block의 head 평균 CLS attention으로 고정한다.
6H Block마다 attention을 다시 측정하는 정책은 이번 실험에 포함하지 않는다.
따라서 유지율 변경과 중요도 재평가 효과를 섞지 않는다.

3H와 entropy routing(τ=1.0), 공식 EMA 가중치, 전처리, 표본 순서, FP32/TF32 조건을 통일한다.
학습 없이 공식 EMA에서 시작하며, 기존 파인튜닝 수치와 직접적인 개선을 주장하지 않는다.
모든 이미지를 각 6H 설정으로 계산해 같은 표본의 correctness를 저장하고,
3H entropy로 routing을 사후 적용한다. 정확도 측정시간은 조기 종료 추론시간이 아니다.
latency 모드는 실제로 확실한 이미지의 6H를 생략한다.

유지율은 **원래 패치 196개에 대한 절대 비율**이다. CLS는 별도로 항상 유지한다.
Block 진입 전 토큰을 gather로 제거하고, 원래 공간 순서를 보존한다.
제거한 토큰은 다시 넣지 않으므로 스케줄은 단조 감소한다.
고정 비율 r이면 모든 Block의 입력 수는 `1 + round(196*r)`이다.

## 사전에 정한 설정

마지막 유지율 r은 25%, 50%, 75%다. 6H의 12개 Block에 다음을 적용한다.

| 설정 | Block 1–4 | Block 5–8 | Block 9–12 |
| --- | --- | --- | --- |
| full | 100% | 100% | 100% |
| uniform_r | r | r | r |
| staged_r | 100% | (100%+r)/2 | r |
| late_r | 100% | 100% | r |
| linear_r | 1번의 100%에서 12번의 r까지 선형 감소 (Block마다 다름) | ← | ← |

`late_r`는 기존 full-k=8 Drop과 대응하는 재현 대조군이다.
동일한 마지막 유지율이면 앞 Block에서 많이 계산한 설정이 유리할 수 있어,
각 가변 스케줄에 **전체 6H 해석적 MACs가 가장 가까운 uniform_nK**도 추가한다.
패치 수가 정수라 MACs가 완전히 같지는 않다. 그 차이를 `uniform_cost_delta_percent`에 표시한다.

MACs는 기존 `plot_h1_diagnosis.py`의 계산 관례와 같다.
Block 폭 d, 토큰 n이면 `12*n*d² + 2*n²*d`이고 Block마다 실제 입력 수를 대입한다.
patch embedding, 3H→6H projection, 분류 head도 포함한다.
정렬/gather·norm·softmax·entropy 비용은 제외한 해석적 값이며 profiler 결과가 아니다.
평균 MACs는 `3H_MACs + routing_fraction * 6H_MACs`다.

## Colab 설정

1. [노트북 열기](https://colab.research.google.com/github/hee4040/OnDevice/blob/codex/layer-wise-drop/notebooks/layerwise_drop.ipynb).
2. 필요하면 **Drive에 사본 저장**. 런타임 유형을 **T4 GPU**로 변경한다.
3. 설치 셀 실행: 별도 Python 3.10 환경과 기존 torch 2.5.1/torchvision 0.20.1을 사용한다.
   Colab 기본 Python 버전과 관계없이 같은 모델 환경을 사용하도록 분리했다.
4. Drive 연결을 직접 승인한다. 데이터는 `MyDrive/OnDevice/data/imagenet1k/`에 저장한다.
5. [ImageNet 데이터셋](https://huggingface.co/datasets/ILSVRC/imagenet-1k) 접근 신청/이용 조건을 직접 확인하고,
   승인 계정의 읽기 토큰을 Colab Secrets `HF_TOKEN` 또는 숨김 입력칸에 입력한다.
   토큰은 채팅·코드·Git에 저장하지 않는다. 승인되지 않은 계정은 다운로드할 수 없다.
6. validation 50,000장만 다운로드한다. Drive와 Colab 로컬 디스크의 여유 공간을 확인한다.
   Drive에는 이미지 약 6.4GB에 실행 결과가 추가된다. 15GB 이상 여유를 권장한다.
7. 로컬 디스크로 복사한 뒤 smoke 64장 → 본실험 50,000장 → latency 1,000장 순으로 실행한다.

검증 데이터는 클래스별 50장·1,000개 숫자 폴더 구성을 검사한다.
데이터 다운로드는 기존 shard manifest로 이어받는다.
평가는 배치마다 Drive에 원자적으로 저장하고 동일 명령 재실행 시 완료 배치를 건너뛴다.
재개 시 소스·가중치·데이터 목록·인수·환경이 달라지면 중단한다. 변경했다면 새 output 폴더를 사용한다.
런타임 리셋 후에는 설치·Drive 연결·인증·로컬 복사를 다시 수행한다.

## 검증과 출력

새 100% 경로와 공식 full 6H logits 일치를 각 배치에서 검증한다.
첫 배치에서는 고정 25/50/75%가 기존 Drop 구현과 같은지도 비교한다.
단위 검사는 토큰 수·개별 이미지별 토큰 정체성·CLS 보존·기존 Drop 일치·MACs 대조군·routing을 검사한다.

`MyDrive/OnDevice/results/layerwise-drop-v1/` 아래에 다음이 생성된다.

- `smoke/`: 64장 실행 확인. 성능 결론에 사용하지 않는다.
- `accuracy/summary.json`, `summary.csv`: 전체/어려운 이미지 Top-1, 평균 GMACs, 가까운 uniform과의 paired 차이.
- `accuracy/environment.json`: 두 저장소 commit/상태, 소스·checkpoint SHA256, 데이터 revision, 전처리, GPU, 버전, 표본 인덱스.
- `accuracy/batch_*.pt`: 이미지 인덱스/정답/entropy/설정별 correctness. logits 전체는 저장하지 않는다.
- `latency-gpu/`: 전체 추론 평균·p50·p95와 이미지별 반복 측정 원자료. batch 1, 기본 3회 반복.
- `figures/`: 실제 측정 결과로 그린 Pareto와 토큰 수 스케줄.

가까운 uniform과의 정확도 차이에 대한 95% 구간은 **사전에 고정한 각 비교의 paired 정규근사 구간**이다.
다중 비교 보정이나 가장 좋은 설정 선택에 대한 보정은 아니다.
같은 validation에서 많은 설정을 비교하므로 최고 성능을 독립 검증 결과로 표현하지 않는다.
threshold 예산 대조군은 entropy 순서에서 가장 가까운 정수 이미지 수를 routing한다.
경계 entropy 동률이 있으면 실제 단일 threshold로 도달할 수 없는 점이므로 별도 flag를 남긴다.

latency는 실제 3H→gate→필요한 6H 전체를 측정하며 attention 수집·정렬/gather·GPU 동기화를 포함한다.
읽기/전처리/H2D 전송은 제외한다. 각 설정의 두 분기를 warmup하고 동일 seed 표본을 사용한다.
GPU의 실행 순서/열/런타임 부하 영향이 남을 수 있으며, Colab GPU 시간은 휴대폰 측정치가 아니다.
CPU 1 thread 비교는 노트북의 선택 셀로 실행할 수 있다.

결과 공유 시 summary/environment/그림만 검토해 Git에 추가한다.
ImageNet 이미지, Hugging Face 토큰, 가중치, 대량 중간 배치는 커밋하지 않는다.
