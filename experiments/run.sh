#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  cat <<'EOF'
Usage: bash experiments/run.sh GROUP RUN_DIRECTORY [--smoke]
Groups: main, h1, drop-ft, allocation, layerwise, evo, evo-train, hardware, hardware-legacy, imagenet-v2, target80-replay
Set IMAGENET_V2_DIR to the extracted matched-frequency directory for imagenet-v2.
Run allocation in the same directory before imagenet-v2, and drop-ft before evo-train.
Smoke mode (64-image wiring check) is available for h1, layerwise and evo.
EOF
}
if [[ ${1:-} == --help || $# -eq 0 ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi
group=$1
mkdir -p "$2"
out=$(cd "$2" && pwd)
py="$root/.venv/bin/python"
case "$out/" in
  "$root"/results/*) ;;
  *) printf 'Choose a run directory under %s/results/\n' "$root" >&2; exit 2 ;;
esac
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
weights="$root/checkpoints/kd_student.pth"
samples=()
case "${3:-}" in
  '') ;;
  --smoke)
    case "$group" in h1|layerwise|evo) samples=(--samples 64);; *) usage >&2; exit 2;; esac
    out="$out/smoke"
    mkdir -p "$out"
    ;;
  *) usage >&2; exit 2;;
esac
cd "$root"

stage() {
  local source=$1
  local workspace="$out/source-$source"
  mkdir -p "$workspace"
  # Keep upstream scripts' relative paths and generated reports inside the run directory.
  cp -R "$root/experiments/$source/." "$workspace/"
  for directory in .venv .cache checkpoints data third_party; do
    if [[ ! -L "$workspace/$directory" ]]; then
      ln -s "$root/$directory" "$workspace/$directory"
    fi
  done
  mkdir -p "$workspace/results"
  cd "$workspace"
}

integration() {
  local name=$1
  shift
  "$py" -m efficient_vit "$@" --output "$out/$name.json"
}

case "$group" in
  main)
    integration validate-real-fp32 validate --real-images
    integration validate-fp16 validate --precision fp16 --real-images
    integration accuracy-original-fp32 evaluate --official --runtime native --policies full
    integration latency-original-fp32 benchmark --official --runtime native --policies full
    for precision in fp32 fp16; do
      integration "accuracy-official-$precision" evaluate --official --precision "$precision" --policies full
      integration "latency-official-$precision" benchmark --official --precision "$precision" --policies full
      integration "accuracy-kd-$precision" evaluate --precision "$precision"
      integration "latency-kd-$precision" benchmark --precision "$precision"
    done
    ;;
  h1)
    stage h1
    "$py" baseline/evaluate_h1_diagnosis.py "${samples[@]}"
    "$py" kd.py "$weights" "${samples[@]}"
    if (( ${#samples[@]} == 0 )); then
      "$py" baseline/evaluate_drop.py
      "$py" baseline/evaluate_drop.py --weights "$weights"
    fi
    ;;
  drop-ft)
    stage drop-ft
    official="$PWD/results/rtx4090-official"
    if [[ ! -s "$official/per_image.pt" ]]; then
      "$py" baseline/evaluate_drop.py --workers 4 --output-dir "$official"
    fi
    "$py" baseline/run_drop_ft_study.py --official-eval "$official"
    reference="$PWD/results/20261106T124010.390968Z-drop-eval-official"
    if [[ ! -e "$reference" && ! -L "$reference" ]]; then
      ln -s "$official" "$reference"
    fi
    "$py" baseline/run_partial_ft_study.py
    retrained="$PWD/results/drop-ft-20261106/kd/train/student.pth"
    cd "$root"
    integration accuracy-retrained-kd-fp32 evaluate --weights "$retrained"
    integration latency-retrained-kd-fp32 benchmark --weights "$retrained"
    ;;
  allocation)
    "$py" -m experiments.allocation.evaluate --output "$out/allocation.json"
    "$py" -m experiments.allocation.benchmark "$out/allocation.json" "$out/allocation-latency.json"
    ;;
  layerwise)
    stage layerwise
    "$py" baseline/evaluate_layerwise_drop.py --output "$out/layerwise-official-accuracy" "${samples[@]}"
    "$py" kd.py "$weights" --output "$out/layerwise-kd-accuracy" "${samples[@]}"
    if (( ${#samples[@]} == 0 )); then
      "$py" baseline/evaluate_layerwise_drop.py --mode latency --samples 1000 --output "$out/layerwise-official-latency"
      "$py" kd.py "$weights" --mode latency --samples 1000 --output "$out/layerwise-kd-latency"
    fi
    ;;
  evo)
    stage drop-ft
    timing=()
    if (( ${#samples[@]} == 0 )); then timing=(--latency-samples 1000); fi
    "$py" baseline/evaluate_evolution.py --weights "$weights" --ratios .5 --output-dir "$out/evo" "${samples[@]}" "${timing[@]}"
    ;;
  evo-train)
    stage drop-ft
    "$py" baseline/run_evolution_study.py
    "$py" baseline/run_evolution_warm_start.py
    ;;
  hardware)
    integration kd-eager-latency benchmark --runtime eager --policies economy kd50 balanced accuracy full
    for control in reembed eager-strict compiled-strict graph-strict; do
      "$py" -m experiments.hardware.controls "$control" "$out"
    done
    ;;
  hardware-legacy)
    stage h1
    "$py" baseline/optimize_latency.py
    ;;
  target80-replay)
    stage allocation/target80
    package=baseline/reports/2026-11-08/target80/replay/numeric_artifacts.zip
    revision=$(cat SOURCE)
    git -C "$root" cat-file -e "$revision:$package"
    git -C "$root" show "$revision:$package" > "$package"
    "$py" scripts/restore_target80_artifacts.py
    "$py" baseline/target80_gate_search.py \
      --source results/20261107T091527.251697Z-adaptive-drop --output "$out/target80-replay"
    ;;
  imagenet-v2)
    v2_data=$(realpath "${IMAGENET_V2_DIR:?Set IMAGENET_V2_DIR to the extracted matched-frequency directory}")
    for mode in official kd; do
      "$py" -m experiments.generalization.imagenet_v2 "$mode" "$v2_data" "$out/allocation.json" "$out"
    done
    ;;
  *) usage >&2; exit 2;;
esac
