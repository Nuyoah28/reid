"""Read public dataset pointers; does not bypass gated/download authorization."""
import concurrent.futures
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URLS={
 'github_occluded':'https://api.github.com/search/repositories?q=Occluded_REID+dataset&per_page=5',
 'hf_occluded':'https://huggingface.co/api/datasets?search=occluded&limit=100',
 'bpbreid_dataset':'https://raw.githubusercontent.com/VlSomers/bpbreid/main/torchreid/data/datasets/image/occluded_reid.py',
 'original_occluded':'https://raw.githubusercontent.com/tine17/occluded-reid/master/README.md',
 'prior_counterfactual':'https://api.openalex.org/works?search=counterfactual%20occlusion%20person%20re-identification&per-page=5',
 'prior_leakage':'https://api.openalex.org/works?search=occluder%20feature%20pollution%20person%20re-identification&per-page=5',
}
def get(item):
 key,url=item
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'reid-research/1.0'})
  data=urllib.request.urlopen(request,timeout=25).read().decode()
  dest=ROOT/'tmp'/'sources';dest.mkdir(parents=True,exist_ok=True);(dest/(key+'.txt')).write_text(data,encoding='utf-8')
  if key=='github_occluded':return key,[(x['full_name'],x['description']) for x in json.loads(data).get('items',[])]
  if key=='hf_occluded':return key,[x['id'] for x in json.loads(data)]
  if key.startswith('prior_'):return key,[(x['title'],x.get('doi')) for x in json.loads(data).get('results',[])]
  return key,data[:10000]
 except Exception as exc:return key,str(exc)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor() as pool:
  for result in pool.map(get,URLS.items()):print(json.dumps(result,ensure_ascii=True),flush=True)
