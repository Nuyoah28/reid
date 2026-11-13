"""Save a short excerpt from the official CCF AI venue list for venue planning."""
import html
import json
import re
import urllib.request
from pathlib import Path

url='https://www.ccf.org.cn/Academic_Evaluation/AI/'
with urllib.request.urlopen(url,timeout=30) as response:
    source=response.read().decode('utf-8')
source=re.sub(r'<script\b[^>]*>.*?</script>',' ',source,flags=re.S|re.I)
source=re.sub(r'<style\b[^>]*>.*?</style>',' ',source,flags=re.S|re.I)
text=html.unescape(re.sub('<[^>]+>',' ',source))
text=re.sub(r'\s+',' ',text)
matches={}
for term in ['ACCV','BMVC','IJCNN','PRCV','PRICAI']:
    at=text.find(term)
    matches[term]=text[max(0,at-140):at+240] if at>=0 else None
root=Path(__file__).resolve().parents[1]
verified=all(matches.values())
(root/'research'/'ccf_venue_check.json').write_text(json.dumps({'url':url,'retrieval_date':'2026-11-13','status':'page_retrieved' if verified else 'not_verified_captcha_or_missing_content','excerpts':matches,'note':'Do not assert venue category without checking the current official list. The supplied catalogue is not an independent verification.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'verified':verified,'excerpts':matches}),flush=True)
