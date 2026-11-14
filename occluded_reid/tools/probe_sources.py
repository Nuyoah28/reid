import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS={
 'bpbreid': 'https://raw.githubusercontent.com/VlSomers/bpbreid/main/README.md',
 'horeid': 'https://raw.githubusercontent.com/DFCVLab/2021-CVPR-HOReID/master/README.md',
 'osnet_paper': 'https://arxiv.org/pdf/1905.00953',
 'market_mirror': 'https://hf-mirror.com/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/Market-1501-v15.09.15.zip?download=true',
 'osnet_gdrive': 'https://drive.usercontent.google.com/download?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY&export=download&confirm=t',
 'market_fresh': 'https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/Market-1501-v15.09.15.zip?download=true&fresh='+str(time.time()),
 'weights_fresh': 'https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/osnet_x1_0_imagenet.pth?download=true&fresh='+str(time.time()),
}

def get(item):
 key,url=item
 try:
  request=urllib.request.Request(url)
  with urllib.request.urlopen(request,timeout=20) as response:
   binary=response.read(4096 if key.startswith(('market','weights','osnet_gdrive')) else -1)
   if key in ('bpbreid','horeid','osnet_paper'):
    dest=ROOT/'tmp'/'sources';dest.mkdir(parents=True,exist_ok=True)
    (dest/(key+('.pdf' if key.endswith('paper') else '.md'))).write_bytes(binary)
   return {'source':key,'status':response.status,'bytes_read':len(binary),'type':response.headers.get('Content-Type'),'head_hex':binary[:12].hex()}
 except Exception as exc:return {'source':key,'error':str(exc)}

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
  for result in pool.map(get,TASKS.items()):print(json.dumps(result),flush=True)
