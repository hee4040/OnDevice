"""Experiment 2 on a Kaggle T4: end-to-end eager latency, to check the stage-sum estimate.

The team measurement (evaluate.py --mode latency) and the 11/3 whole-inference timing
(measure_latency.py --mode pipeline, GPU only: CPU times on Kaggle say nothing about a device).
ImageNet validation comes from the ILSVRC competition data, laid out as in fp16.py.
"""
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path



def find(name):  # dataset mounts are symlinks, which Path.rglob does not follow
    for folder, dirs, files in os.walk("/kaggle/input", followlinks=True):
        dirs[:] = [d for d in dirs if d != "ILSVRC"]
        if name in files or name in dirs:
            return Path(folder) / name
    raise FileNotFoundError(name)


print(sorted(os.walk("/kaggle/input", followlinks=True).__next__()[1]))
try:
    tarfile.open(find("ondevice.tar")).extractall("/tmp")
except FileNotFoundError:  # Kaggle unpacked the archive
    shutil.copytree(find("OnDevice"), "/tmp/OnDevice")
root = Path("/tmp/OnDevice")
Path("/kaggle/working/results").mkdir(exist_ok=True)
(root / "results").symlink_to("/kaggle/working/results")
competition = find("LOC_val_solution.csv").parent
index = {s: i for i, s in enumerate(sorted(l.split()[0] for l in open(competition / "LOC_synset_mapping.txt")))}
validation = Path("/tmp/data/imagenet1k/validation")
for i in range(1000):
    (validation / f"{i:04d}").mkdir(parents=True)
images = competition / "ILSVRC/Data/CLS-LOC/val"
for line in open(competition / "LOC_val_solution.csv").read().splitlines()[1:]:
    name, boxes = line.split(",", 1)
    (validation / f"{index[boxes.split()[0]]:04d}" / f"{name}.JPEG").symlink_to(images / f"{name}.JPEG")
data = ["--data-dir", "/tmp/data/imagenet1k"]
subprocess.run([sys.executable, "baseline/evaluate.py", "--mode", "latency", "--samples", "1000", "--warmup", "50",
                "--verify-samples", "100", *data], cwd=root)  # the pipeline timing below runs even if this fails
subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'baseline'); import measure_latency as m; "
                "m.SETTINGS = m.SETTINGS[:1]; m.main()", "--mode", "pipeline", "--images", "1000", *data],
               cwd=root, check=True)
