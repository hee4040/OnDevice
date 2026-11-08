"""Experiment 2 on a Kaggle T4: stage timings of each runtime (optimize_latency.py). No ImageNet needed."""
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
subprocess.run(["nvidia-smi"])
subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"])
subprocess.run([sys.executable, "baseline/optimize_latency.py"], cwd=root, check=True)
