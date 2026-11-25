#!/usr/bin/env python3
"""Experiment 2: does the same drop Top-K model get faster on the GPU with better execution?

Times each stage of the inference (3H, 3H with the CLS attention drop needs, full 6H, drop
Top-K 6H at 25/50/75%) with float32 weights (TF32 allowed, as in measure_latency.py) and
float16 weights, each run four ways: eager PyTorch, CUDA Graphs (one manual capture per stage),
torch.compile, and torch.compile with CUDA Graphs (mode="reduce-overhead"). Batch 1 for all;
batch 8 and 32 for three settings, to see from which batch size fewer tokens save GPU time.

Drop gathers the kept tokens with fixed-shape indices (topk, then sort) instead of
mask.nonzero(), which waits for the GPU and cannot be captured in a graph; the logits are
checked against the mask path first. Timing loops are short with a pause between them, since
the shared lab PC reset under long full GPU load. The reported results ran on a Kaggle T4
(scripts/kaggle_exp2/latency.py).
"""
import argparse
from datetime import datetime, timezone
import json
import time

import torch

from evaluate_drop import NARROW_P, ROOT, STAGES, WIDE_P, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from measure_latency import timed
from selective_refinement import forward_narrow_with_attention, make_update_mask

RATIOS = [0.25, 0.5, 0.75]
RUNTIMES = ["eager", "CUDA Graphs", "torch.compile", "torch.compile + CUDA Graphs"]
SWEEP = [("float32", "eager"), ("float16", "eager"), ("float16", "torch.compile + CUDA Graphs")]


def drop_index(scores, ratio):
    """CLS, then the Top-K patches in sequence order: what mask.nonzero() gives, at a fixed shape."""
    top = scores[:, 1:].topk(round((scores.shape[1] - 1) * ratio), dim=1).indices.sort(dim=1).values + 1
    return torch.cat([torch.zeros_like(top[:, :1]), top], dim=1)


def drop_logits(model, x0, keep):
    model.p = WIDE_P
    x = x0.gather(1, keep.unsqueeze(-1).expand(-1, -1, x0.shape[-1]))
    for block in model.blocks:
        x = block(x, WIDE_P)
    return model.forward_head(model.norm(x, int(WIDE_P * model.embed_dim)))


def check(model):
    """The fixed-shape drop path must give the logits of the mask path used for every accuracy number."""
    x = torch.randn(8, 3, 224, 224, generator=torch.Generator().manual_seed(1)).to(next(model.parameters()).device)
    features, _, attention = forward_narrow_with_attention(model, x, STAGES)
    x0 = wide_input(model, x, features)
    for r in RATIOS:
        torch.testing.assert_close(drop_logits(model, x0, drop_index(attention, r)),
                                   wide_logits(model, x0, make_update_mask(attention, r, "topk")))


def stages(model, x):
    features, _, attention = forward_narrow_with_attention(model, x, STAGES)

    def narrow():
        model.p = NARROW_P
        return model.entropy(model.forward_head(model.forward_features(x, NARROW_P)))

    def narrow_attention():
        _, logits, captured = forward_narrow_with_attention(model, x, STAGES)
        return model.entropy(logits), captured

    out = {"3H": narrow, "3H + attention": narrow_attention,
           "6H full": lambda: wide_logits(model, wide_input(model, x, features))}
    for r in RATIOS:
        out[f"6H drop Top-K {round(100 * r)}%"] = (
            lambda r=r: drop_logits(model, wide_input(model, x, features), drop_index(attention, r)))
    return out


def prepare(fn, runtime):
    if runtime == "eager":
        return fn
    if runtime == "CUDA Graphs":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # capture needs a warmed-up side stream
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph.replay
    return torch.compile(fn, mode="reduce-overhead" if "Graphs" in runtime else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--pause", type=float, default=3.0, help="Idle seconds between timings")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.distributions.Distribution.set_default_validate_args(False)  # the entropy's argument check waits for the GPU
    model = load_model(args.repo.resolve(), args.checkpoint.resolve())
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-optimize-latency")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    rows = []
    with torch.no_grad():
        check(model)
        for precision in ["float32", "float16"]:
            model = model.to(getattr(torch, precision))
            for batch in [1, 8, 32]:
                x = torch.randn(batch, 3, 224, 224, generator=torch.Generator().manual_seed(0))
                x = x.to("cuda", getattr(torch, precision))
                for runtime in RUNTIMES:
                    if batch > 1 and (precision, runtime) not in SWEEP:
                        continue
                    for name, fn in stages(model, x).items():
                        torch._dynamo.reset()  # fresh compile per stage, no shared recompile limit
                        start = time.perf_counter()
                        try:
                            run = prepare(fn, runtime)
                            run()
                            torch.cuda.synchronize()
                            setup = time.perf_counter() - start
                            row = {"setup_s": setup, **timed(run, "cuda", 10, args.runs)}
                        except Exception as error:  # what cannot be captured or compiled is a result too
                            row = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
                        rows.append({"precision": precision, "batch": batch, "runtime": runtime, "stage": name, **row})
                        print(json.dumps(rows[-1]), flush=True)
                        (output / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
                        time.sleep(args.pause)
    environment = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "float32": "TF32 allowed", "float16": "model.half(), float16 input",
                   "measurement": "synchronized wall time of one stage, median of runs; excludes preprocessing and H2D"}
    (output / "summary.json").write_text(json.dumps({"environment": environment, "rows": rows}, indent=2) + "\n")
    print("Done", flush=True)


if __name__ == "__main__":
    main()
