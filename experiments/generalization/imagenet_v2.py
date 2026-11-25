import json
import sys
from pathlib import Path
import torch
from efficient_vit import __main__ as evaluation
from efficient_vit.policy import Policy, PRESETS

mode, source_dir, allocation_path, output_dir = sys.argv[1:]
source, out = Path(source_dir), Path(output_dir)
def dataset_for(native, args):
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset
    from timm.data import create_transform, resolve_data_config
    dataset = ImageFolder(source, transform=create_transform(
        **resolve_data_config({}, model=native, use_test_size=True), is_training=False))
    assert sorted(map(int, dataset.classes)) == list(range(1000))
    assert len(dataset) == 10000
    dataset.target_transform = lambda index: int(dataset.classes[index])
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    return Subset(dataset, indices), indices

evaluation.dataset_for = dataset_for
allocation = json.loads(Path(allocation_path).read_text())
for name, row in allocation["budgets"][1]["results"].items():
    gate, boundary = row["gate"], row["boundary"]
    PRESETS[name] = Policy(gate, (boundary,), (.25, .5)) if boundary > gate else Policy(gate, (), (.5,))
official = mode == "official"
options = ["--official", "--policies", "full"] if official else ["--policies", "fixed", "gate_only", "ratio_only", "joint", "balanced", "accuracy"]
sys.argv = ["efficient_vit", "evaluate", *options, "--output", str(out / ("official.json" if official else "kd.json"))]
evaluation.main()
