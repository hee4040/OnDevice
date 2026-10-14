#!/usr/bin/env python3
"""Fetch the official, standard (not 800-epoch) EMA export and verify strict loading."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = "NCPS/thinkingvit_deit-3h-6h-imagenet1k"


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(ROOT / "third_party/ThinkingViT"))
    import torch
    import timm
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file
    if not Path(timm.__file__).resolve().is_relative_to(ROOT / "third_party/ThinkingViT"):
        raise RuntimeError("The official repository's custom timm must be imported")
    directory = ROOT / "checkpoints"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "thinkingvit_3h6h_ema.pth"
    metadata = directory / "thinkingvit_3h6h_ema.json"
    if checkpoint.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != saved["sha256"]:
            raise RuntimeError("Existing checkpoint hash mismatch")
        print(f"Verified existing checkpoint: {checkpoint}")
        return
    info = HfApi().model_info(REPO)
    files = {f.rfilename for f in info.siblings}
    filename = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    original = hf_hub_download(REPO, filename, revision=info.sha)
    weights = load_file(original) if filename.endswith(".safetensors") else torch.load(original, map_location="cpu", weights_only=True)
    model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=[3, 6])
    model.load_state_dict(weights, strict=True)
    temporary = checkpoint.with_suffix(".partial")
    torch.save({"state_dict_ema": weights}, temporary)
    temporary.replace(checkpoint)
    metadata.write_text(json.dumps({"repository": REPO, "revision": info.sha, "source_file": filename,
                                   "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(f"Official EMA weights prepared: {checkpoint}")


if __name__ == "__main__":
    main()
