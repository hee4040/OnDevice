import sys
from pathlib import Path

import torch
from efficient_vit import __main__ as evaluation
from efficient_vit.runtime import Engine

control = sys.argv[1]
out = Path(sys.argv[2])
original = evaluation.Engine

def evaluation_engine(model, runtime):
    evaluation.Engine = original
    if control.endswith("-strict"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    engine = Engine(model, runtime)
    if control.startswith("compiled"):
        engine.narrow_call = torch.compile(engine.narrow_call, dynamic=False)
        engine.wide_calls[.5] = torch.compile(engine.wide_calls[.5], dynamic=False)
    elif control == "reembed":
        narrow = engine.narrow_call
        wide = engine.wide_calls[.5]
        previous = None
        def remember(images):
            nonlocal previous
            previous = images
            return narrow(images)
        engine.narrow_call = remember
        engine.wide_calls[.5] = lambda _embedding, features, scores: wide(model.embed(previous), features, scores)
    return engine

evaluation.Engine = evaluation_engine
runtime = "graph" if control == "graph-strict" else "eager"
sys.argv = ["efficient_vit", "benchmark", "--runtime", runtime, "--policies", "kd50",
            "--output", str(out / f"kd-{control}-latency.json")]
evaluation.main()
