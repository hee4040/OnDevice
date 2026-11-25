#!/usr/bin/env python3
"""Download authorized ImageNet-1K Parquet shards and preserve original image bytes.

No dataset terms are accepted automatically. Log in after accepting the dataset
conditions at https://huggingface.co/datasets/ILSVRC/imagenet-1k.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = "ILSVRC/imagenet-1k"
COUNTS = {"train": 1281167, "validation": 50000}


def is_transient_download_error(error):
    """Hub can wrap a metadata timeout in LocalEntryNotFoundError."""
    from requests.exceptions import ConnectionError, HTTPError, SSLError, Timeout, ChunkedEncodingError

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, HTTPError) and error.response is not None:
            status = error.response.status_code
            return status in {408, 429} or 500 <= status < 600
        if isinstance(error, SSLError):
            return False  # Certificate/configuration errors need intervention.
        if isinstance(error, (Timeout, ConnectionError, ChunkedEncodingError)):
            return True
        error = error.__cause__ or error.__context__
    return False


def retry_download(operation, description, retries=8):
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as error:
            if not is_transient_download_error(error):
                raise
            if attempt == retries:
                print(f"{description}: retries exhausted; completed shards and partial downloads "
                      "are preserved. Run the same command again to resume.", flush=True)
                raise
            delay = min(5 * 2 ** attempt, 60)
            # Do not print exception messages containing signed download URLs.
            print(f"{description}: temporary network failure ({type(error).__name__}); "
                  f"retry {attempt + 1}/{retries} in {delay}s", flush=True)
            time.sleep(delay)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def split_files(files, split):
    selected = sorted(f for f in files if re.fullmatch(
        rf"data/{split}-\d+-of-\d+\.parquet", f))
    if not selected:
        raise ValueError(f"No {split} Parquet shards found; inspect the upstream layout.")
    totals = {int(Path(f).stem.rsplit("-", 1)[1]) for f in selected}
    indices = {int(Path(f).stem.split("-")[1]) for f in selected}
    if totals != {len(selected)} or indices != set(range(len(selected))):
        raise ValueError(f"Incomplete shard list for {split}")
    return selected


def check_counts(split_dir, expected):
    counts = {}
    directories = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if directories != [f"{label:04d}" for label in range(1000)]:
        raise ValueError(f"{split_dir}: expected exactly the class folders 0000..0999")
    for name in directories:
        counts[name] = sum(1 for p in (split_dir / name).iterdir()
                           if p.is_file() and p.suffix.lower() in {".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"})
    if sum(counts.values()) != expected or any(n == 0 for n in counts.values()):
        raise ValueError(f"Incorrect image counts: {sum(counts.values())}, expected {expected}")
    if split_dir.name == "validation" and set(counts.values()) != {50}:
        raise ValueError("ImageNet validation must contain exactly 50 images per class")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--splits", nargs="+", choices=list(COUNTS), default=["validation", "train"])
    parser.add_argument("--revision", default=None, help="HF commit; defaults to the saved commit after the first run")
    parser.add_argument("--retries", type=int, default=8, help="Additional attempts per network operation (default: 8)")
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    # huggingface_hub reads these values at import time.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    metadata_timeout = int(os.environ["HF_HUB_ETAG_TIMEOUT"])
    if metadata_timeout <= 0 or int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]) <= 0:
        parser.error("HF_HUB_ETAG_TIMEOUT and HF_HUB_DOWNLOAD_TIMEOUT must be positive integers")
    from huggingface_hub import HfApi, hf_hub_download, get_token
    import pyarrow.parquet as pq
    from PIL import Image

    if not get_token():
        raise SystemExit("First accept the ImageNet conditions on Hugging Face and run huggingface-cli login. Do not send tokens in chat.")
    args.output.mkdir(parents=True, exist_ok=True)
    state_dir = args.output / ".preparation"
    state_dir.mkdir(exist_ok=True)
    lock_path = state_dir / "source.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    revision = args.revision or (lock["revision"] if lock else "main")
    api = HfApi()
    info = retry_download(
        lambda: api.dataset_info(REPO, revision=revision, files_metadata=True, token=True,
                                 timeout=metadata_timeout),
        "Dataset metadata", args.retries)
    if lock and lock["revision"] != info.sha:
        raise SystemExit("A different dataset revision is already present. Use a new output directory.")
    selected = {s: split_files([f.rfilename for f in info.siblings], s) for s in dict.fromkeys(args.splits)}
    sizes = {f.rfilename: f.size for f in info.siblings}
    # HEAD a gated shard before creating a source lock or claiming access.
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    first = next(iter(selected.values()))[0]
    retry_download(
        lambda: get_hf_file_metadata(hf_hub_url(REPO, first, repo_type="dataset", revision=info.sha),
                                     token=True, timeout=metadata_timeout),
        "Dataset access check", args.retries)
    write_json(lock_path, {"repository": REPO, "revision": info.sha,
                           "label_mapping": "HF integer labels preserved as zero-padded folder names"})
    for split, shards in selected.items():
        destination = args.output / split
        destination.mkdir(exist_ok=True)
        for label in range(1000):
            (destination / f"{label:04d}").mkdir(exist_ok=True)
        for index, filename in enumerate(shards):
            marker = state_dir / (Path(filename).stem + ".json")
            if marker.exists():
                manifest = json.loads(marker.read_text())
                if all((destination / item["path"]).is_file() and
                       (destination / item["path"]).stat().st_size == item["bytes"]
                       for item in manifest["images"]):
                    print(f"[{split} {index + 1}/{len(shards)}] already prepared", flush=True)
                    continue
            reserve = 5 * 1024**3 + 3 * (sizes[filename] or 1024**3)
            if shutil.disk_usage(args.output).free < reserve:
                raise SystemExit("Insufficient free disk space for the next shard; completed shards are preserved.")
            print(f"[{split} {index + 1}/{len(shards)}] {filename}", flush=True)
            shard = retry_download(
                lambda: hf_hub_download(REPO, filename, repo_type="dataset", revision=info.sha,
                                        token=True, etag_timeout=metadata_timeout),
                filename, args.retries)
            images = []
            table = pq.ParquetFile(shard)
            for batch in table.iter_batches(batch_size=32, columns=["image", "label"]):
                for row in batch.to_pylist():
                    label = row["label"]
                    if not isinstance(label, int) or not 0 <= label < 1000:
                        raise ValueError(f"Invalid class index: {label}")
                    raw = row["image"]["bytes"]
                    if not raw:
                        raise ValueError("Missing embedded image bytes")
                    with Image.open(io.BytesIO(raw)) as im:
                        extension = {"JPEG": ".JPEG", "PNG": ".png", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff", "WEBP": ".webp"}.get(im.format)
                        if extension is None:
                            raise ValueError(f"Unsupported image format {im.format}")
                        im.load()  # Detect decoding failures before declaring the shard ready.
                    relative = Path(f"{label:04d}") / f"{Path(filename).stem}-{len(images):06d}{extension}"
                    target = destination / relative
                    temporary = target.with_suffix(target.suffix + ".partial")
                    temporary.write_bytes(raw)  # No JPEG re-encoding or resizing.
                    temporary.replace(target)
                    images.append({"path": str(relative), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "source_path": row["image"].get("path")})
            write_json(marker, {"shard": filename, "revision": info.sha, "images": images})
        counts = check_counts(destination, COUNTS[split])
        write_json(state_dir / f"{split}-complete.json", {"revision": info.sha, "count": sum(counts.values()), "classes": counts})
        print(f"{split}: verified {sum(counts.values())} images / 1000 classes", flush=True)
    print("Requested splits prepared. Source Parquet files remain in .cache/huggingface for resuming.")


if __name__ == "__main__":
    main()
