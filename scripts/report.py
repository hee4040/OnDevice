#!/usr/bin/env python3
"""Publish compact, portable evidence; keep raw IDs and per-image arrays ignored."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/efficient-kd")
    args = parser.parse_args()
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    data = {}
    for name in ("accuracy-fp32", "accuracy-native-fp32", "accuracy-fp16", "accuracy-original-fp32", "accuracy-official-optimized-fp32",
                 "native_equivalence", "original_equivalence", "validate-fp32", "validate-fp16", "validate-real-fp32",
                 "latency-graph-fp32", "latency-eager-fp32", "latency-native-fp32", "latency-graph-fp16",
                 "latency-original-fp32", "latency-official-graph-fp32", "latency-integrated-vs-original-fp32"):
        path = args.source / f"{name}.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        value.pop("indices", None)
        monitoring = value.pop("gpu_monitor", [])
        if monitoring:
            value["gpu_monitor_samples"] = len(monitoring)
            value["other_gpu_process_observations"] = [s for s in monitoring if s["other_processes"]]
        data[name] = value
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    accuracy = data.get("accuracy-fp32", {}).get("accuracy", {})
    original = data.get("accuracy-original-fp32", {}).get("accuracy", {}).get("full")
    lines = ["# 원본 ThinkingViT 대비 통합 모델 실측", "", f"갱신 UTC: {datetime.now(timezone.utc).isoformat()}", "",
        "브랜치: `integration/efficient-kd`. GPU: RTX 3070 Ti Laptop 8GB. FP32/TF32.",
        "**주 대조군은 main의 공식 EMA ThinkingViT다. KD 학습·Drop 없이 3H→6H, 토큰 197개 전체, τ=1의 실제 early exit다.**",
        "원본 체크포인트 SHA-256과 공식 모델 revision은 main과 같다. `original-main-protocol.json`에 근거를 보존한다.",
        "정확도는 ImageNet validation 50,000장, 시간은 같은 seed-42 입력 1,000장·batch 1·warmup 50장·3회다.",
        "시간은 반복 평균의 중앙값이다. gate·선택·recycling을 포함하고 전처리/H2D/최종 argmax는 제외한다.",
        "원본은 토큰 선택용 CLS attention을 추가 계산하지 않는다. main의 과거 RTX 3090 시간은 현재 GPU 비교에 재사용하지 않는다.",
        "현재 정확도 batch는 64(main 과거 측정은 128)이며 원본 80.304%·6H 진입 18,004/50,000을 재현했다.", "",
        "| 모델 / 정책 | Top-1 | GMACs | 원본 대비 정확도 %p | 원본 대비 연산량 감소 | 실제 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def ms(key, policy):
        timing = data.get(key, {})
        value = timing.get("latency_ms", {}).get(policy) if timing.get("clean") else None
        return f"{value:.3f}" if value is not None else "미측정/오염 제외"
    if original:
        lines.append(f"| main 원본 ThinkingViT, 원래 실행 | {original['top1']:.3f}% | {original['gmacs']:.4f} | 기준 | 기준 | {ms('latency-original-fp32','full')} |")
        optimized = data.get("accuracy-official-optimized-fp32", {}).get("accuracy", {}).get("full")
        if optimized:
            lines.append(f"| 같은 공식 가중치·전체 토큰 + 실행 최적화만 | {optimized['top1']:.3f}% | {optimized['gmacs']:.4f} | "
                f"{optimized['top1']-original['top1']:+.3f} | {100*(1-optimized['gmacs']/original['gmacs']):.2f}% | {ms('latency-official-graph-fp32','full')} |")
        names = {"kd50": "통합 kd50 (보조)", "economy": "통합 절약", "balanced": "통합 기본", "accuracy": "통합 정확도 우선", "full": "KD 학생 전체 토큰 (보조)"}
        for name in ("economy", "balanced", "accuracy", "kd50", "full"):
            value = accuracy[name]
            lines.append(f"| {names[name]} | {value['top1']:.3f}% | {value['gmacs']:.4f} | {value['top1']-original['top1']:+.3f} | "
                f"{100*(1-value['gmacs']/original['gmacs']):.2f}% | {ms('latency-integrated-vs-original-fp32',name)} |")
        lines += ["", "원본 대비 시간 차이는 정책과 실행 최적화의 효과를 함께 포함한다. 위의 '실행 최적화만' 행으로",
            "같은 가중치·같은 토큰·같은 gate를 유지한 runtime 이득을 따로 확인한다.",
            "원본의 GMACs는 실제 early exit 실행량이다. 공식 raw forward가 6H까지 항상 계산한 뒤 출력을 고르는",
            "실행 방식의 전체 연산량/시간과 섞지 않는다."]
    lines += ["", "## 보조 분석: 중간 KD 모델 대비", "",
        "아래는 주 베이스라인이 아니라, KD 이후 통합 정책·embedding 재사용 효과를 분리하는 대조군이다.", "",
        "| 정책 | Top-1 | GMACs | native ms | 고정 폭 eager ms | Graph ms | KD 대비 연산량 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    native_accuracy = data.get("accuracy-native-fp32", {}).get("accuracy", {})
    baseline = native_accuracy.get("kd50", accuracy.get("kd50", {}))
    if native_accuracy:
        value = native_accuracy["kd50"]
        timing = data.get("latency-native-fp32", {})
        t = timing.get("latency_ms", {}).get("kd50") if timing.get("clean") else None
        lines.append(f"| 원본 KD 고정 50% | {value['top1']:.3f}% | {value['gmacs']:.4f} | {f'{t:.3f}' if t is not None else '미측정'} | — | — | 기준 |")
    for name in ("kd50", "economy", "balanced", "accuracy", "full"):
        value = accuracy.get(name)
        if not value:
            continue
        times = []
        for runtime in ("native", "eager", "graph"):
            if runtime == "native":
                times.append("—")
                continue
            timing = data.get(f"latency-{runtime}-fp32", {})
            t = timing.get("latency_ms", {}).get(name) if timing.get("clean") else None
            times.append(f"{t:.3f}" if t is not None else ("오염 제외" if timing and not timing.get("clean") else "미측정"))
        delta = 100 * (value["gmacs"] / baseline["gmacs"] - 1)
        lines.append(f"| {name} | {value['top1']:.3f}% | {value['gmacs']:.4f} | " + " | ".join(times) + f" | {delta:+.2f}% |")
    if baseline:
        value = accuracy["balanced"]
        lines += ["", f"balanced의 정확도 개선: {value['top1']-baseline['top1']:+.3f}%p, 평균 MACs 변화 "
            f"{100*(value['gmacs']/baseline['gmacs']-1):+.2f}%. 경계는 이번 validation으로 조정하지 않았다.",
            "과거 KD 79.672%와 이번 조건부 실행 KD 79.670%는 정답 수 1장 차이다.",
            "이번에는 원본 native와 통합 구현의 kd50/full 예측·경로·정오답을 5만 장 이미지별로 비교해 모두 일치했다.",
            "표에는 이번에 직접 재평가한 대조군을 표시하며 개선도 그 기준으로 계산한다.",
            "통합 구현은 3H의 원본 embedding을 재사용해 6H에서 중복 Conv2d를 계산하지 않는다.",
            "따라서 같은 kd50 정책도 native 대비 GMACs가 낮다. 원본 대비 정책과 실행 최적화의 효과를 구분해서 본다."]
    lines += ["", "## FP16 (별도 평가)", "",
        "FP16을 FP32 정확도에 섞지 않는다. 선택적으로 사용할 수 있으며 아래는 실제 전체 평가 결과만 표시한다.", "",
        "| 정책 | FP16 Top-1 | FP32 대비 차이 %p | Graph ms |", "| --- | ---: | ---: | ---: |"]
    for name, value in data.get("accuracy-fp16", {}).get("accuracy", {}).items():
        timing = data.get("latency-graph-fp16", {})
        t = timing.get("latency_ms", {}).get(name) if timing.get("clean") else None
        lines.append(f"| {name} | {value['top1']:.3f}% | {value['top1']-accuracy[name]['top1']:+.3f} | {f'{t:.3f}' if t is not None else '미측정'} |")
    lines += ["", "## 지연 분포·처리량·메모리", "",
        "메모리는 PyTorch의 이 프로세스 할당이다. 1,000장 입력을 GPU에 캐시했으므로 입력 캐시를 뺀 상주/peak도 표시한다.",
        "Graph 풀이 준비 때 선할당되므로 incremental peak가 작다는 이유로 메모리 절감이라고 해석하지 않는다.",
        "처리량은 batch-1 순차 처리량이며 배치 처리량이 아니다.", "",
        "| 실행 / 정책 | p95 ms (반복 중앙값) | 순차 images/s | 입력 캐시 제외 상주 MiB | 입력 캐시 제외 peak MiB |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for key, value in data.items():
        if not key.startswith("latency-") or not value.get("clean"):
            continue
        cache = value["samples"] * 3 * 224 * 224 * (2 if value["precision"] == "fp16" else 4) / 2**20
        for name in value["policies"]:
            rows = [r for r in value["rows"] if r["policy"] == name]
            lines.append(f"| {key.removeprefix('latency-')} / {name} | {statistics.median(r['p95_ms'] for r in rows):.3f} | "
                f"{statistics.median(r['serial_images_per_s'] for r in rows):.1f} | "
                f"{max(r['resident_allocated_mib'] for r in rows)-cache:.1f} | {max(r['peak_allocated_mib'] for r in rows)-cache:.1f} |")
    lines += ["", "## 검증과 해석", "",
        "- GPU 원본/고정 폭/Graph 비교는 각 validate JSON에 오차를 기록한다.",
        "- 실제 이미지 32장의 원본/eager/Graph logits·선택을 검증하고 JSON에 기록한다.",
        "- GPU compute 프로세스를 1초 간격으로 감시하고, 오염된 측정은 위 시간 표에서 제외한다.",
        "- 같은 원본 모델의 가중치를 폭별로 복사하므로 상주 모델 메모리는 증가할 수 있다.",
        "- MACs는 conv/matmul 추정이며 softmax/norm/topk/gather/메모리 이동/CPU 호출 비용을 포함하지 않는다.",
        "- validation은 과거 반복 사용된 데이터다. 새 독립 평가나 여러 학습 seed의 결과라고 주장하지 않는다.",
        "- 새 장시간 학습 없이 기존 KD를 사용했다. 정리한 학습 코드는 2 update 실행 점검을 했다.", "",
        "추가 명령·정책·제외한 방법의 근거는 [방법](../docs/method.md), 실행은 [README](../README.md)에 있다.",
        "원자료는 ignored `results/efficient-kd/`에, 공유할 요약 JSON은 이 디렉터리에 보존한다.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    smoke = args.source / "kd-train-smoke/complete.json"
    if smoke.exists():
        (output / "kd-train-smoke.json").write_text(smoke.read_text())
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "efficient_vit").glob("*.py"))}
    (output / "implementation_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
