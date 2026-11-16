#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-data.txt
if [[ "${1:-}" == "--data-only" ]]; then
    echo 'Data tools installed. Login with: HF_HOME="$PWD/.cache/huggingface" .venv/bin/huggingface-cli login'
    exit 0
fi
EXPECTED_REVISION=9254c4950d3159937ad682592d58a2be69708c42
if [[ ! -d third_party/ThinkingViT/.git ]]; then
    git clone https://github.com/ds-kiel/ThinkingViT.git third_party/ThinkingViT
    git -C third_party/ThinkingViT checkout --detach "$EXPECTED_REVISION"
fi
# Pin the native model used to validate the fixed-width conversion.
if [[ "$(git -C third_party/ThinkingViT rev-parse HEAD)" != "$EXPECTED_REVISION" ]]; then
    echo "ThinkingViT must use revision $EXPECTED_REVISION; current checkout was left unchanged." >&2
    exit 1
fi
git -C third_party/ThinkingViT rev-parse HEAD > third_party/ThinkingViT.commit
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements-installed.txt
.venv/bin/python scripts/prepare_weights.py
