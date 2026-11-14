"""Record runtime versions without modifying the existing conda environment."""
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
import torchvision
import numpy
from PIL import __version__ as pil_version

ROOT=Path(__file__).resolve().parents[1]
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
record={'python_executable':sys.executable,'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'pillow':pil_version,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':gpu,'conda_environment':'mmi-unet','package_changes':'None. Research runner uses the existing torch/torchvision/numpy/Pillow stack; optional upstream scipy/timm/gdown are not imported.'}
(ROOT/'research'/'environment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
(ROOT/'research'/'environment-pip-freeze.txt').write_text(freeze,encoding='utf-8')
print(json.dumps(record),flush=True)
