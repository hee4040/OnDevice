"""One consolidated, atomically updated record of the token-evolution studies."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clock(value):
    return datetime.fromisoformat(value).astimezone(KST).strftime("%m-%d %H:%M:%S") if value else "진행 중"


def write_report(directory):
    original = read(directory / "study.json") or {}
    followup = read(directory / "warm_start_study.json") or {"status": "추가 실험 예약 전"}
    lines = ["# Global CLS · Slow-fast 전체 실험 기록", "",
        f"갱신: {datetime.now(KST).isoformat(timespec='seconds')} (KST)", "",
        f"기존 실험 상태: **{original.get('status', '확인 중')}**", "",
        f"이어받기 실험 상태: **{followup['status']}**", "",
        "## 실험 구성과 현재 판단", "",
        "기존 전체 KD의 유지율 50%·entropy τ=1 정확도 79.672%를 기준으로 비교한다.",
        "기존 KD 가중치에 선택 방식만 바꾼 사전 평가와, 변경 구조를 적용한 재학습 결과를 구분한다.",
        "Global CLS 재학습 결과는 79.678%로 기준보다 +0.006%p(5만 장 중 3장) 높다.",
        "단일 seed에서 사실상 같은 수준이며 뚜렷한 정확도 개선으로 주장하지 않는다.", "",
        "| 실험 | 시작 가중치 | 학습 구조 | 추가 학습 | 누적 KD 학습 |",
        "| --- | --- | --- | ---: | ---: |",
        "| 기존 KD | 공식 EMA | 마지막 3H CLS로 고정 Top-K Drop | 3에폭 | 3에폭 |",
        "| 선택 방식 사전 비교 | 완료된 기존 KD | 재학습 없이 네 추론 방식 적용 | 0 | 3에폭 |",
        "| global_cls | 공식 EMA | Global CLS + 고정 Top-K Drop | 3에폭 | 3에폭 |",
        "| evo_schedule | 공식 EMA | Global CLS + slow-fast + layer-to-stage | 3에폭 | 3에폭 |",
        "| global_to_evo (추가) | 완료된 global_cls의 student.pth | Global CLS + slow-fast + layer-to-stage | 3에폭 | 6에폭 |", "",
        "evo_schedule과 global_to_evo 모두 **Global CLS를 포함**한다. Global CLS 없이 slow-fast만 학습한 실험은 없다.",
        "global_to_evo는 Global CLS 가중치만 학생에 복사한다. Teacher는 계속 공식 EMA 전체 토큰 모델이며,",
        "optimizer·warmup·cosine·RNG·step은 새 학습으로 시작한다. 이전 optimizer 상태를 이어받지 않는다.",
        "기존 evo_schedule 학습·평가를 먼저 끝내고 추가 실험을 순차 실행한다.",
        "추가 실험은 총 6에폭이므로 초기 가중치 효과와 학습량 효과를 분리한 비교가 아니다.", "",
        "## 구현한 방법", "",
        "1. **Global CLS:** 새 학습 토큰을 추가하지 않는다. 3H의 첫 블록 CLS attention으로 초기화하고,",
        "   이후 12블록에서 head 평균 attention을 `g = 0.5*g + 0.5*a`로 누적한다. 그 순위로 6H 토큰을 고른다.",
        "2. **Slow-fast:** 6H에서 선택 패치는 직접 갱신하고, 비선택 패치는 global attention의 정규화 가중합으로",
        "   대표 토큰 하나를 만든다. CLS·선택 패치·대표 토큰만 블록에 넣고 대표 토큰의 **출력−입력 잔차**를",
        "   비선택 패치마다 더한다. 원래 위치를 보존해 이후 재선택이 가능하며 attention EMA는 선택 패치만 갱신한다.",
        "3. **Layer-to-stage:** slow-fast 학습 첫 2에폭은 매 블록에서 재선택하고 마지막 1에폭은 4블록마다 재선택한다.",
        "   각 블록의 특징·attention 갱신은 계속 수행한다. 기본 추론은 4블록 단위(evo_stage)다.",
        "   같은 가중치에서 블록별 재선택(evo_layer)도 평가한다. 별도의 layer-only 학습 실험은 아니다.", "",
        "3H는 197개 토큰을 유지하고, 6H 재활용 입력을 만든 뒤 압축한다. 대표 토큰도 기존 Drop과 같은 예산에 포함한다.",
        "유지율 50%일 때 기존은 CLS 1+패치 98, slow-fast는 CLS 1+패치 97+대표 1로 둘 다 블록 입력이 99개다.",
        "대표 생성·gather/scatter·점수 추출 비용은 추가되므로 토큰 수 일치가 실제 시간 일치를 뜻하지 않는다.",
        "논문 Eq.4–7을 ThinkingViT에 적용한 변형이다. 공식 DeiT 코드는 앞 4블록을 dense로 처리하고 placeholder를",
        "그대로 보존하는 변형을 사용한다. 이번 구현은 3H 점수로 6H 첫 블록부터 압축하고 논문 수식의 잔차를 전파한다.", "",
        "## 공통 학습·평가 설정", "",
        "| 항목 | 설정 |", "| --- | --- |",
        "| 데이터 | ImageNet-1K train 1,281,167장 / validation 50,000장 |",
        "| 손실 | 0.5 KL(3H) + 0.5 KL(압축 6H), T=1; CE·Full·Feature 추가 없음 |",
        "| 학습 | 전체 backbone, 3에폭·30,027 update, 유효 batch 128 / micro 32, seed 42 |",
        "| optimizer | AdamW lr 5e-5, weight decay 0, warmup 500, cosine, gradient clip 1 |",
        "| 유지율 | 학습 시 batch마다 25/50% 선택, 재학습 후 평가 25/50/75% |",
        "| 정밀도 | BF16 학습 / FP32·TF32 허용 평가 |",
        "| 평가 | seed-42 validation 순서, τ=1 고정, 마지막 checkpoint; validation으로 epoch 선택 안 함 |",
        "| 모델·GPU | ThinkingViT 3H→6H, 224×224, RTX 3070 Ti Laptop 8GB |", "",
        "유지율은 버린 비율이 아니라 남기는 패치 예산 비율이다. routed 정확도는 entropy≥1이면 압축 6H,",
        "그 외에는 3H 정답을 사용한다. 전체 6H 단독 정확도와 이 지표는 다르다.", "",
        "## 학습 진행", "", "| 실험 | update | 완료율 | 상태 |", "| --- | ---: | ---: | --- |"]
    for name in ("global_cls", "evo_schedule", "global_to_evo"):
        base = directory / name / "train"
        arguments = read(base / "arguments.json") or {}
        complete = read(base / "complete.json")
        log = read(base / "train_log.json") or []
        total = arguments.get("steps", 30027)
        step = complete["steps"] if complete else (log[-1]["step"] if log else 0)
        status = "학습 완료" if complete else ("학습 중" if arguments else "대기")
        lines.append(f"| {name} | {step:,}/{total:,} | {100 * step / total:.1f}% | {status} |")
    lines += ["", "## 재학습 전 사전 비교", "", "완료된 기존 KD 가중치를 그대로 사용한 5만 장 평가다.", ""]
    def accuracy_table(name):
        result = read(directory / name / "eval/summary.json")
        if not result:
            return ["아직 평가 결과 없음.", ""]
        rows = [f"표본 {result['samples']:,}장 · 평가 완료 {clock(result.get('finished'))} (KST)", "",
                "| 추론 경로 | 해당 경로 단독 Top-1 | τ=1 최종 Top-1 |", "| --- | ---: | ---: |"]
        for key, value in result["top1"].items():
            rows.append(f"| {key.removesuffix('_correct')} | {value['all']:.3f}% | {value['threshold_1']:.3f}% |")
        return rows + [""]
    lines += accuracy_table("existing_kd")
    lines += ["## 구조 적용 후 재학습 결과", ""]
    for name in ("global_cls", "evo_schedule", "global_to_evo"):
        lines += [f"### {name}", ""] + accuracy_table(name)
    from evolution_compute import write_compute, plot_tradeoff
    compute = write_compute(directory)
    plotted_models = read(directory / "accuracy_compute_tradeoff.models.json")
    complete_models = sorted(set(r["model"] for r in compute["rows"]))
    if compute["rows"] and plotted_models != complete_models:
        plot_tradeoff(directory)
    lines += ["## 정확도·연산량 트레이드오프", "",
        "주 비교는 원본 전체 토큰 3H→6H·τ=1 대비다. GMACs는 이미지당 평균 곱셈-누산 횟수(10억 단위)이며,",
        "여기서는 1 MAC을 곱셈+누산 한 쌍으로 센다(보통 2 FLOPs). 실제 추론 시간과 분리해 보고한다.",
        "원본 대비 정확도 하락을 허용하면서 GMACs를 줄이는지, 같은 GMACs에서 정확도가 높은지를 확인한다.",
        "Global CLS/slow-fast는 같은 유지율에서 기존 Drop과 블록 토큰 예산이 같아 주 연산량은 거의 같다.",
        "목표는 해당 예산에서 정확도 보존이다. GMACs 절감의 대부분은 토큰 수 감소에서 발생한다.", "",
        "| 모델 / 경로 | 유지율 | Top-1 | 평균 GMACs | 원본 대비 연산량 감소 | 정확도 차이 %p | 깨끗한 batch-1 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for point in compute["rows"]:
        latency = f"{point['latency_ms']:.3f}" if point["latency_ms"] is not None else "미측정"
        lines.append(f"| {point['model']} / {point['method']} | {100*point['ratio']:.0f}% | {point['top1']:.3f}% | "
            f"{point['gmacs']:.4f} | {point['mac_reduction_percent']:.2f}% | {point['accuracy_delta_pp']:+.3f} | {latency} |")
    lines += ["", "이 수치는 conv·선형층·attention 행렬곱·재활용 projection·선택용 CLS QK·대표 가중합의 분석적 계산이다.",
        "norm/softmax/GELU·elementwise 갱신·정렬·gather/scatter·메모리 이동·CPU 호출 비용은 GMACs에서 제외한다.",
        "따라서 연산량이 줄어도 실제 GPU 시간이 늘 수 있다. GPU 대기·메모리 병목을 분리하려면 profiler가 필요하며,",
        "현재 자료만으로 병목의 원인을 확정하지 않는다. 지연 평균/p95, batch throughput, peak VRAM도 별도 지표다.",
        "기존 저장소 수치(원본 약 2.915 GMACs)는 3H patch embedding을 192채널로 가정했다.",
        "실제 구현은 384채널 embedding을 만든 뒤 첫 블록에서 자르므로 여기서는 모든 경로에 이를 보정했다.",
        "GMACs 수치가 약 0.029 증가한 것은 계산 기준 보정이며 모델 변경 때문이 아니다.",
        "CPU에서 실제 모듈 입출력 형태를 추적해 3H core 1,282,584,576 MACs와 50% 6H core 2,265,169,920 MACs를 확인했다.",
        "증거: `compute_shape_verification.json`. GPU 속도를 CPU에서 측정한 값이 아니다.",
        "원자료: `compute_tradeoff.json`. 그림의 회색 곡선은 원본 모델의 threshold 탐색으로, 같은 연산량에서의 대조군이다.",
        "threshold 곡선은 같은 validation에서 얻은 사후 탐색이며 별도 calibration으로 선택한 배포 성능은 아니다.", ""]
    figure = directory / "accuracy_compute_tradeoff.png"
    if figure.exists():
        lines += [f"![정확도 대 평균 GMACs]({figure})", "",
                  "그림은 생성 시점에 평가가 완료된 모델만 포함한다. 위 표는 후속 평가 완료 시 자동 추가된다.", ""]
    lines += ["## GPU 추론 시간 재측정", ""]
    for filename in ("latency-clean-now.json", "latency-clean-final.json"):
        measurement = read(directory / filename)
        if not measurement:
            continue
        lines += [f"### {filename}", "", f"상태: **{measurement['status']}**", "",
            "같은 세션·같은 seed-42 validation 1,000장·batch 1·τ=1·유지율 50%·FP32/TF32·CPU thread 1.",
            "50장 warmup, 방법 순서를 섞어 3회 반복, 각 반복 평균의 중앙값과 범위를 표시한다.",
            "전처리/H2D/최종 argmax 제외, 분기·선택 부가 연산 포함. full 대조군은 선택용 attention을 추출하지 않는다.",
            "1초 간격으로 다른 GPU compute 프로세스와 정지 상태, 온도·사용률·전력·클럭을 기록한다.",
            "진행 중인 모델은 최종 가중치가 없으므로 재측정 표에 포함하지 않는다.", ""]
        if measurement.get("clean"):
            lines += ["GPU 공유 작업이 관측되지 않은 측정이다. 메모리에 남은 학습 프로세스는 SIGSTOP 상태로 검증했다.", "",
                "| 가중치 / 경로 | 평균의 중앙값 ms | 반복 평균 범위 ms |", "| --- | ---: | --- |"]
            for case, value in measurement.get("aggregate", {}).items():
                lines.append(f"| {case} | {value['median_mean_ms']:.3f} | {value['min_mean_ms']:.3f}–{value['max_mean_ms']:.3f} |")
            lines += [""]
        else:
            lines += ["아직 공유 작업 없는 최종 측정이 확인되지 않아 확정 시간으로 사용하지 않는다.", ""]
        if measurement.get("training_resumed"):
            lines += [f"기존 학습 재개: {clock(measurement['training_resumed'])} (KST).", ""]
    lines += ["## 과거 GPU 추론 시간 기록", "",
        "오염된 과거 측정은 보존하며, 위 재측정이 완료되면 위 수치를 우선 사용한다.",
        "동일 모델·동일 seed 순서의 1,000장, warmup 50장, batch 1, τ=1, 유지율 50%.",
        "방법 순서를 순환하고 CUDA 동기화 wall-clock으로 측정한다. 전처리·H2D 제외, 분기 동기화 포함.",
        "사전 비교(existing_kd)는 다른 분류기 학습과 GPU를 공유했고 full 대조군도 마지막 CLS attention을 추출했다.",
        "따라서 사전 비교와 이후 평가의 절대 시간을 직접 비교하지 않는다. 재학습 가중치 평가는 full 대조군의",
        "선택용 attention 추출을 생략한다. 동시 프로세스가 관측되면 전용 GPU에서의 속도로 해석하지 않는다.", "",
        "| 가중치 | 경로 | 평균 ms | 측정 전후 다른 GPU 프로세스 |", "| --- | --- | ---: | --- |"]
    for name in ("existing_kd", "global_cls", "evo_schedule", "global_to_evo"):
        result = read(directory / name / "eval/summary.json") or {}
        timing = result.get("latency", {})
        shared = "있음(별도 관측 기록)" if name == "existing_kd" else (
            "관측됨" if timing.get("other_gpu_processes_before") or timing.get("other_gpu_processes_after") else "없음(전후 확인)")
        for method, ms in timing.get("mean_ms", {}).items():
            lines.append(f"| {name} | {method} | {ms:.3f} | {shared} |")
    lines += ["", "## 실행 시각·소요 시간", "", "시각은 KST, 시간은 프로세스 준비·학습/평가·저장을 포함한다.", "",
        "| 단계 | 시작 KST | 종료 KST | 경과 | 상태 |", "| --- | --- | --- | --- | --- |"]
    for stage in original.get("stages", []) + followup.get("stages", []):
        start = datetime.fromisoformat(stage["started"])
        end = datetime.fromisoformat(stage["finished"]) if stage.get("finished") else datetime.now(timezone.utc)
        seconds = max(0, int((end - start).total_seconds()))
        duration = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
        lines.append(f"| {stage['name']} | {clock(stage['started'])} | {clock(stage.get('finished'))} | {duration} | {stage['status']} |")
    lines += ["", "## 검증·해석·재실행", "",
        "기존 단위 테스트 40개와 실제 GPU에서 기존 Drop 출력 일치·전체 토큰 경로 일치·BF16 역전파 검증을 통과했다.",
        "추가한 학생 초기화는 공식 Teacher를 복사한 뒤 수행하므로 Teacher 가중치는 바꾸지 않는다.",
        "이어받기 경로는 실제 train 이미지로 2 update·역전파·체크포인트 저장을 확인했다.",
        "검증 시 초기 가중치 SHA-256은 Global CLS 평가 summary의 가중치 SHA-256과 일치했다.",
        "각 결과는 같은 validation을 재사용한 단일 seed 실험이다. 작은 정확도 차이를 확정적 개선으로 보지 않는다.",
        "추가 실험의 개선은 학습 초기화와 누적 6에폭의 영향을 함께 포함한다. 다른 seed·동일 학습량 대조군은 수행하지 않았다.", "",
        "```bash", ".venv/bin/python baseline/run_evolution_warm_start.py", "```", "",
        "위 명령은 기존 runner의 잠금이 해제되고 기존 slow-fast 평가가 완료된 뒤 추가 실험을 실행한다.",
        "완료 단계는 재실행하지 않으며 중단 학습은 optimizer·schedule·RNG·batch 위치를 복원한다.",
        "worker augmentation의 bit 단위 재현은 보장하지 않는다. Git commit·push 없음.", "",
        f"원자료: `{directory}`. 기존 실행 명령·종료 코드: `study.json`, 추가 실행: `warm_start_study.json`.",
        "각 실험의 `train/arguments.json`, `train/train_log.json`, `train/complete.json`, `eval/summary.json`,",
        "`eval/per_image.pt`에 설정·진행·완료·이미지별 결과를 보존한다.",
        "`global_to_evo/train/arguments.json`에는 초기 가중치 경로와 SHA-256을 기록한다.", "",
        "[Evo-ViT 논문](https://arxiv.org/abs/2108.01390) ·",
        "[공식 코드](https://github.com/YifanXu74/Evo-ViT/blob/main/deit/evo_deit.py)", ""]
    target = ROOT / "baseline/evolution_experiments.md"
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(target)
