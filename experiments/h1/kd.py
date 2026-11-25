import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "baseline"))
import evaluate_h1_diagnosis as evaluation

weights = Path(sys.argv.pop(1))
sys.path.insert(0, str((evaluation.ROOT / "third_party/ThinkingViT").resolve()))
import validate

prepare = validate.RLDropping

def load_weights(model, *args, **kwargs):
    prepare(model, *args, **kwargs)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("student", state))

validate.RLDropping = load_weights
configs = [row for row in evaluation.experiment_configs()
           if row["k"] == 0 and row["ratio"] == .5 and
           (row["mode"] == "drop" or row["policy"] == "topk")]
configs.append({**next(row for row in configs if row["mode"] == "drop"),
                "name": "drop_k0_lownorm_50", "policy": "lownorm"})
evaluation.experiment_configs = lambda: configs
previous = set((evaluation.ROOT / "results").glob("*-h1-diagnosis"))
evaluation.main()
output, = set((evaluation.ROOT / "results").glob("*-h1-diagnosis")) - previous
path = output / "environment.json"
environment = json.loads(path.read_text())
environment["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
path.write_text(json.dumps(environment, indent=2) + "\n")
(evaluation.ROOT / "results" / "h1-kd-summary.json").write_text((output / "summary.json").read_text())
