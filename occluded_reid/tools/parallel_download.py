"""Resumable bounded HTTP Range downloader; verifies complete asset SHA256."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from download_assets import ASSETS,BASE,ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('asset',choices=['market','imagenet','occluded']);p.add_argument('--workers',type=int,default=12);args=p.parse_args()
 if args.asset=='occluded':
  name='Occluded_REID.zip';folder='data';size=41937342;expected='1c8854e02db775da6a51399df4b7e02c299804850dff9985451d610fc8c6d57d'
  url='https://raw.githubusercontent.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/61e54da27fc1999dd68ce5681f251220beb0fc64/Occluded_REID.zip'
 else:
  name,expected,folder=ASSETS[args.asset];size={'market':152733771,'imagenet':10910553}[args.asset];url=BASE+name
 dest=ROOT/folder/name;dest.parent.mkdir(exist_ok=True)
 cache=ROOT/'tmp'/'download_chunks'/args.asset;cache.mkdir(parents=True,exist_ok=True)
 chunksize=512*1024
 def part(i):
  lo=i*chunksize;hi=min(size,lo+chunksize)-1;out=cache/f'{i:04d}.bin'
  if out.exists() and out.stat().st_size==hi-lo+1:return i
  for attempt in range(4):
   try:
    fresh=url+('?' if '?' not in url else '&')+f'part={i}&fresh={time.time()}'
    request=urllib.request.Request(fresh,headers={'Range':f'bytes={lo}-{hi}','User-Agent':'reid-research'})
    with urllib.request.urlopen(request,timeout=35) as response:
     if response.status!=206:raise RuntimeError(f'Range not honored: {response.status}')
     data=response.read(hi-lo+2)
    if len(data)!=hi-lo+1:raise RuntimeError(f'Incomplete range {len(data)}')
    out.write_bytes(data);return i
   except Exception as exc:
    if attempt==3:raise RuntimeError(f'Chunk {i}: {exc}') from exc
  return i
 total=(size+chunksize-1)//chunksize;completed=0;last=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(part,i) for i in range(total)]
  for future in concurrent.futures.as_completed(futures):
   future.result();completed+=1
   if time.monotonic()-last>5 or completed==total:
    print(f'{args.asset}: {completed}/{total} chunks ({completed/total*100:.1f}%)',flush=True);last=time.monotonic()
 with dest.with_suffix(dest.suffix+'.parallel').open('wb') as output:
  for i in range(total):output.write((cache/f'{i:04d}.bin').read_bytes())
 actual=hashlib.sha256(dest.with_suffix(dest.suffix+'.parallel').read_bytes()).hexdigest()
 if expected and actual!=expected:raise RuntimeError(f'Hash mismatch {actual}')
 dest.with_suffix(dest.suffix+'.parallel').replace(dest)
 print(f'Completed {dest.name}, SHA256 {actual}',flush=True)

if __name__=='__main__':main()
