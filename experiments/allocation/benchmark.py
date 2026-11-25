"""Benchmark the four policies selected at the baseline budget."""
import json
import sys
from pathlib import Path

from efficient_vit.__main__ import main
from efficient_vit.policy import PRESETS, Policy

source, output = map(Path, sys.argv[1:])
budgets = json.loads(source.read_text())["budgets"]
baseline = next(budget for budget in budgets if budget["multiplier"] == 1.0)
for name, row in baseline["results"].items():
    gate, boundary = row["gate"], row["boundary"]
    PRESETS[name] = Policy(gate, (boundary,), (.25, .5)) if boundary > gate else Policy(gate, (), (.5,))

sys.argv = ["efficient_vit", "benchmark", "--policies", *baseline["results"], "--output", str(output)]
main()
