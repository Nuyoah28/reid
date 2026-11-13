"""Download public research assets with SHA256 verification and safe ZIP extraction."""
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/b5e654a472251f193f78b6ad0af73f74e577a474/'
ASSETS={
 'market':('Market-1501-v15.09.15.zip','416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96','data'),
 'imagenet':('osnet_x1_0_imagenet.pth','fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79','weights'),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('asset',choices=ASSETS);args=p.parse_args()
 name,expected,folder=ASSETS[args.asset];dest=ROOT/folder/name;dest.parent.mkdir(exist_ok=True)
 if not dest.exists():
  part=dest.with_suffix(dest.suffix+'.part');last=time.monotonic()
  # A fresh request avoids caching an expired redirect from an LFS mirror.
  url=BASE+name+'?download=true&fresh='+str(time.time())
  with urllib.request.urlopen(url,timeout=60) as response, part.open('wb') as output:
   size=int(response.headers.get('Content-Length',0));done=0
   while chunk:=response.read(1024*1024):
    output.write(chunk);done+=len(chunk)
    if time.monotonic()-last>5: print(f'{name}: {done/1e6:.1f}/{size/1e6:.1f} MB',flush=True);last=time.monotonic()
  part.replace(dest)
 actual=hashlib.sha256(dest.read_bytes()).hexdigest()
 if actual!=expected: raise RuntimeError(f'SHA256 mismatch: {actual}')
 print(f'Verified {name}: {actual}',flush=True)
 if args.asset=='market':
  with zipfile.ZipFile(dest) as archive:
   root=dest.parent.resolve()
   for member in archive.infolist():
    if not (root/member.filename).resolve().is_relative_to(root):raise RuntimeError('Unsafe ZIP path')
   archive.extractall(root)
  print('Extracted Market-1501',flush=True)
 manifest=ROOT/'research'/('asset_'+args.asset+'.json')
 manifest.write_text(json.dumps({'url':BASE+name,'sha256':actual,'local_path':str(dest.relative_to(ROOT))},indent=2),encoding='utf-8')

if __name__=='__main__':main()
