#!/usr/bin/env python3
"""Does mixing the 3H answer into the 6H answer raise the accuracy of the locked same-budget policy? (CPU only)

For an image the policy sends to 6H, the final answer is argmax((1 - a) * softmax(6H) + a * softmax(3H)). The compute cost
does not change. a is picked on the calibration half and judged on the holdout half (paired bootstrap, 5000 resamples, seed 0).
Input: the validation logit cache written by cache_logits.py. Policy: exit below entropy 0.8, keep 25% up to 1.25, keep 50% above."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="split_ids.json of the adaptive-drop experiment")
    args = ap.parse_args()
    torch.set_num_threads(4)
    c = torch.load(args.val, weights_only=True)
    y, e = c["target"].numpy(), c["entropy"].double().numpy()
    p3 = c["narrow_logits"].float().softmax(1)
    p25, p50 = c["wide_250_logits"].float().softmax(1), c["wide_500_logits"].float().softmax(1)
    split = json.loads(args.split.read_text())
    cal, hold = np.array(sorted(split["calibration"])), np.array(sorted(split["holdout"]))
    low, high = (e >= 0.8) & (e < 1.25), e >= 1.25
    sent, right3 = low | high, p3.argmax(1).numpy() == y

    def correct(a):
        out = p3.argmax(1).numpy().copy()
        out[low] = ((1 - a) * p25 + a * p3).argmax(1).numpy()[low]
        out[high] = ((1 - a) * p50 + a * p3).argmax(1).numpy()[high]
        return out == y

    base = correct(0.0)
    line = lambda name, r: print(f"{name:14s} all {100 * r.mean():.3f}  calibration {100 * r[cal].mean():.3f}  holdout {100 * r[hold].mean():.3f}"  # noqa: E731
                                 f" | harmed {int((sent & right3 & ~r).sum())}  fixed {int((sent & ~right3 & r).sum())}")
    line("6H answer only", base)
    results = {}
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        results[a] = correct(a)
        line(f"mix a={a}", results[a])
    best = max(results, key=lambda a: (results[a][cal].mean(), -a))
    d = results[best][hold].astype(float) - base[hold]
    rng = np.random.default_rng(0)
    means = np.concatenate([d[rng.integers(0, len(d), size=(100, len(d)))].mean(1) for _ in range(50)])
    lo, hi = np.percentile(100 * means, [2.5, 97.5])
    print(f"a picked on calibration = {best} -> holdout change {100 * d.mean():+.3f} pp, 95% [{lo:+.3f}, {hi:+.3f}]")
    right_full = c["wide_1000_logits"].argmax(1).numpy() == y
    all_patches = right3.copy()
    all_patches[sent] = right_full[sent]  # same images sent to 6H, but 6H sees every patch: the most that better drop-robustness could give
    print(f"reference: 3H only {100 * right3.mean():.3f}  every image full 6H {100 * right_full.mean():.3f}"
          f"  same images sent but 6H sees all patches {100 * all_patches.mean():.3f}")


if __name__ == "__main__":
    main()
