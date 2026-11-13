import concurrent.futures
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URLS={
 'original_occluded_readme':'https://raw.githubusercontent.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/master/README.md',
 'original_occluded_tree':'https://api.github.com/repos/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/git/trees/master?recursive=1',
 'cal_paper':'https://arxiv.org/pdf/2108.08728',
 'cal_readme':'https://raw.githubusercontent.com/raoyongming/CAL/master/README.md',
}
def get(item):
 key,url=item
 try:
  data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'reid-research'}),timeout=25).read()
  dest=ROOT/'tmp'/'sources';dest.mkdir(exist_ok=True,parents=True);(dest/(key+('.pdf' if key.endswith('paper') else '.txt'))).write_bytes(data)
  if key.endswith('tree'):
   tree=json.loads(data).get('tree',[]);return key,[(x['path'],x.get('size')) for x in tree][:30],len(tree)
  return key,len(data),data[:8000].decode(errors='replace') if not key.endswith('paper') else ''
 except Exception as e:return key,str(e)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor() as pool:
  for result in pool.map(get,URLS.items()):print(json.dumps(result,ensure_ascii=True),flush=True)
