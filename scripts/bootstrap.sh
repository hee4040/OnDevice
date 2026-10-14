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
if [[ ! -d third_party/ThinkingViT/.git ]]; then
    git clone https://github.com/ds-kiel/ThinkingViT.git third_party/ThinkingViT
fi
# Keep an existing checkout unchanged. Record the exact revision on installation.
git -C third_party/ThinkingViT rev-parse HEAD > third_party/ThinkingViT.commit
.venv/bin/python -m pip install -r third_party/ThinkingViT/requirements.txt
.venv/bin/python -m pip freeze > requirements-installed.txt
.venv/bin/python scripts/prepare_weights.py
