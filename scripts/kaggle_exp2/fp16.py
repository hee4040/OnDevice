"""Experiment 2 on a Kaggle T4: float16 accuracy of the official and fine-tuned models (evaluate_drop.py --fp16).

ImageNet validation comes from the ILSVRC competition data, laid out as the class folders 0000..0999
that evaluate_drop.py expects (sorted synsets = the standard class order, as in the HF labels).
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
for weights in ([], ["--weights", str(root / "student.pth")]):
    subprocess.run([sys.executable, "baseline/evaluate_drop.py", "--fp16", "--workers", "4",
                    "--data-dir", "/tmp/data/imagenet1k", *weights], cwd=root, check=True)
