"""Refresh source checks and render selected supplied PDF pages for human inspection.

Run in the bundled document runtime, not the training conda environment.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'tmp/sources'


def fetch(item):
    key, url = item
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read()
            (DEST/(key+('.pdf' if data.startswith(b'%PDF') else '.html'))).write_bytes(data)
            html = data.decode('utf-8', errors='replace') if not data.startswith(b'%PDF') else ''
            title = re.search(r'<title[^>]*>(.*?)</title>', html, re.S | re.I)
            return {'source': key, 'url': url, 'status': response.status, 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'title': re.sub(r'\s+', ' ', title.group(1)).strip() if title else None,
                    'checked_utc': datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        return {'source': key, 'url': url, 'error': str(exc)}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    selected = json.loads((ROOT/'research/selected_papers.json').read_text(encoding='utf-8'))
    targets = []
    index = []
    pages = {'R0030': [3, 6, 7], 'R0031': [3, 7], 'R0032': [3, 8], 'R0492': [2], 'R0370': [3]}
    for row in selected:
        path = ROOT.parent/'reference-paper'/row['PDF文件名']
        reader = PdfReader(path)
        text = '\n\n'.join(f'PAGE {i+1}\n{page.extract_text()}' for i, page in enumerate(reader.pages))
        (ROOT/'research/paper_text').mkdir(exist_ok=True)
        (ROOT/'research/paper_text'/f"{row['编号']}.txt").write_text(text, encoding='utf-8', errors='replace')
        index.append({'id': row['编号'], 'title_catalogue': row['标题'], 'pages': len(reader.pages),
                      'pdf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scope': 'full text',
                      'filename': path.name})
        if row['编号'] in pages:
            pdf = pdfium.PdfDocument(path)
            for number in pages[row['编号']]:
                pdf[number-1].render(scale=1.4).to_pil().save(DEST/f"{row['编号']}_p{number}.png")
        if row['编号'] in ['R0030', 'R0031', 'R0032']:
            url = row['PDF下载来源'].replace('/papers/', '/html/').replace('_paper.pdf', '_paper.html')
            targets.append((row['编号']+'_cvf', url))
    targets += [('AGW_abstract', 'https://arxiv.org/abs/2001.04193'),
                ('AGW_paper', 'https://arxiv.org/pdf/2001.04193'),
                ('SYSU_official', 'https://isee-ai.cn/project/RGBIRReID.html'),
                ('RegDB_official', 'https://dm.dongguk.edu/link.html')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        checks = list(pool.map(fetch, targets))
    (ROOT/'research/source_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT/'research/reading_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    agw = DEST/'AGW_paper.pdf'
    if agw.exists():
        reader = PdfReader(agw)
        (ROOT/'research/paper_text/AGW.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{p.extract_text()}' for i, p in enumerate(reader.pages)), encoding='utf-8', errors='replace')


if __name__ == '__main__':
    main()
