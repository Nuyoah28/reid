"""Small public literature search audit; discovery is not a novelty proof."""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUERIES=[
    'counterfactual occlusion person re-identification',
    'occluder feature pollution person re-identification',
    'occlusion invariance reliability person re-identification',
    'occluder replacement spatial feature consistency person re-identification',
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'research'/'prior_art_search.json')
    args=p.parse_args()
    report={'provider':'OpenAlex public works search','queries':[],
            'scope':'Top 5 search results per query, a discovery audit only. Ranking/coverage is incomplete; no absence-of-prior-art claim.'}
    for query in QUERIES:
        url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':query,'per-page':5,'select':'id,doi,title,publication_year,primary_location'})
        item={'query':query,'url':url}
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'reid-research'})
            data=json.load(urllib.request.urlopen(request,timeout=30))
            item['results']=[{'title':r['title'],'year':r['publication_year'],'doi':r['doi'],
                              'openalex':r['id'],'landing_page':(r.get('primary_location') or {}).get('landing_page_url')}
                             for r in data['results']]
        except Exception as exc:
            item['error']=str(exc)
        report['queries'].append(item)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
