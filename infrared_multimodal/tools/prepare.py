"""Record the existing environment and fetch the upstream ImageNet initializer only."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import urllib.request
import torch
import torchvision
import numpy
import PIL

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-download', action='store_true')
    args = p.parse_args()
    research = ROOT/'research'
    research.mkdir(exist_ok=True)
    payload = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
               'torch': torch.__version__, 'torchvision': torchvision.__version__,
               'numpy': numpy.__version__, 'pillow': PIL.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               'conda_environment': 'mmi-unet', 'package_changes': 'None; existing packages used.'}
    (research/'environment.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    frozen = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (research/'environment-pip-freeze.txt').write_text(frozen.stdout, encoding='utf-8')
    if not args.no_download:
        path = ROOT/'weights/resnet50-19c8e357.pth'
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix('.part')
            print(f'Downloading ImageNet initialization: {URL}', flush=True)
            with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(2**20):
                    output.write(chunk)
            temporary.replace(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if not checksum.startswith('19c8e357'):
            raise ValueError(f'ImageNet initializer SHA256 prefix mismatch: {checksum}')
        (research/'imagenet_source.json').write_text(json.dumps({'url': URL, 'sha256': checksum,
              'bytes': path.stat().st_size, 'scope': 'ImageNet only; no ReID identity training.'}, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
