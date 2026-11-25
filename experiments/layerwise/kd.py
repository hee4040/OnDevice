import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "baseline"))
import evaluate_layerwise_drop as evaluation

weights = Path(sys.argv.pop(1))
load = evaluation.load_model
evaluation.load_model = lambda repo, checkpoint, device: load(repo, checkpoint, weights=weights, device=device)
# Record the effective KD weights separately from the official EMA.
import hashlib
import json

output = Path(sys.argv[sys.argv.index("--output") + 1])
digest = hashlib.sha256(weights.read_bytes()).hexdigest()
provenance = {"weights_sha256": digest}
record = output / "kd_weights.json"
if output.exists() and any(output.iterdir()):
    if not record.exists() or json.loads(record.read_text()) != provenance:
        raise ValueError("Layer-wise output belongs to another evaluation or checkpoint")
output.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps(provenance, indent=2) + "\n")
evaluation.main()
