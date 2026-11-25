# RTX 4090 실험 코드와 실행 순서

저장소 루트에서 실행한다. [프로젝트 README](../README.md)의 설치 절차로 환경, 공식 EMA, 전달받은 3070 KD 체크포인트, ImageNet-1K validation을 준비한다. Drop FT / Partial FT와 Evo 재학습에는 ImageNet-1K train도 필요하다.

```bash
RUN="results/reproduction-$(date +%Y%m%d-%H%M%S)"
bash experiments/run.sh main "$RUN"
bash experiments/run.sh h1 "$RUN"
bash experiments/run.sh drop-ft "$RUN"
bash experiments/run.sh allocation "$RUN"
bash experiments/run.sh layerwise "$RUN"
bash experiments/run.sh evo "$RUN"
bash experiments/run.sh hardware "$RUN"
```

다른 GPU 연산이 없는 상태에서 지연시간을 측정한다. 기본값은 논문의 평가 규모다. H1, Layer-wise, Evo의 연결만 확인할 때는 마지막에 `--smoke`를 붙일 수 있다. 스모크 출력은 그 실행 디렉터리의 `smoke/`에 기록된다. 실험 단계는 별도로 실행할 수 있으며, 새 측정은 새 실행 디렉터리를 사용한다.

| README 순서 | 명령의 그룹 | 코드와 출처 |
| --- | --- | --- |
| 2.1–2.3 메인 | `main`, `h1`, `allocation` | [통합 실행기](../efficient_vit), [H1](h1), [공동 배분](allocation) |
| 3.1 토큰 처리, 3.2 토큰 선택 | `h1` | [H1 진단·Drop 평가](h1/baseline), `experiment/h1-diagnosis` at `9d8f655` |
| 3.3 Fine-tuning | `drop-ft` | [Drop FT, Partial FT](drop-ft/baseline), `experiment/drop-ft` at `d9e28b6` |
| 3.4 Fixed / Gate-only / Ratio-only / Joint | `allocation` | [예측 생성·세 예산 탐색·CUDA Graph 지연시간](allocation), 기존 `efficient_vit` 추론 및 MAC 산식 사용 |
| 3.5 Layer-wise | `layerwise` | [레이어별 평가](layerwise/baseline), `codex/layerwise-drop-results` at `424b185` |
| 3.6 Global CLS / Evo | `evo` | [Evo 평가](drop-ft/baseline/evaluate_evolution.py), `experiment/drop-ft` at `d9e28b6` |
| 3.7 GPU 실행 | `hardware`, `hardware-legacy` | [통합 실행기 대조군](hardware), [기존 H1 최적화](h1/baseline/optimize_latency.py) |
| 5.1 ImageNet-V2 | `imagenet-v2` | [V2 데이터 연결](generalization/imagenet_v2.py), 기존 `efficient_vit` 평가 사용 |

Global CLS/Evo를 별도로 재학습하려면 `drop-ft`를 마친 **같은 실행 디렉터리**에서 `bash experiments/run.sh evo-train "$RUN"`을 실행한다. 3070 가중치 평가만 필요하면 `evo` 단독으로 실행한다. 기존 H1 하드웨어 스크립트의 캡처·컴파일 오류를 확인하려면 `hardware-legacy`를 별도로 실행한다. 통합 하드웨어 재현은 `main`의 Graph/FP16 기준선과 `hardware`의 Eager·재임베딩·엄격 FP32 비교로 구성된다.

[공개 ImageNet-V2 matched-frequency](https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-matched-frequency.tar.gz) 파일을 풀고, 숫자 클래스 폴더 1,000개가 들어 있는 경로를 지정한다. `allocation`에서 저장한 정책을 같은 실행 디렉터리에서 사용한다.

```bash
IMAGENET_V2_DIR=/path/to/imagenetv2-matched-frequency-format-val \
  bash experiments/run.sh imagenet-v2 "$RUN"
```

이전 Target80의 저장 예측 재계산은 별도 명령이다. 원본 `codex/target80-results`의 숫자 자료 아카이브와 Matplotlib이 필요하다.

```bash
git fetch origin codex/target80-results
.venv/bin/python -m pip install matplotlib
bash experiments/run.sh target80-replay "$RUN"
```

## 원본과 추가 코드

각 `SOURCE`에는 전체 커밋 해시가 있다. 그 디렉터리의 `baseline/` 및 `scripts/`에는 해당 커밋에서 실제로 실행한 진입점과 필요한 의존 파일을 원래 바이트 그대로 가져왔다. H1 진단과 토큰 선택은 동일 구현을 공유한다. Drop FT와 Evo도 학습·모델 로직을 공유하므로 한 폴더에 묶었다. 같은 이름의 의존 파일은 각 원본 버전의 경로와 출처를 유지한다.

이 작업에서 추가한 코드는 실행 순서와 경로를 정리하는 러너, 기존 H1·Layer-wise에 3070 가중치를 연결하는 어댑터, 기존 통합 모델로 Gate-only·Ratio-only·Joint를 비교하는 평가 및 지연시간 실행 코드, 런타임 대조군, V2 데이터 어댑터다. Token Drop, Selective Refinement, Partial FT, Layer-wise Drop, Global CLS, Evo 자체의 구현은 기존 코드다. 공동 배분의 동일 예산 비교는 이전 Target80 재계산과 별도로 이 작업에서 추가한 실험이다.

러너가 원본 묶음을 실행 디렉터리 아래에 복사하므로 원본 스크립트의 상대 경로와 출력 구조를 유지한다. 가중치, 데이터, 대용량 로그는 로컬 파일로 제공한다.
