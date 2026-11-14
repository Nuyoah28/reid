"""Fetch primary DPM preprint and record limited prototype-related source access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DOIS = ['10.1145/3503161.3547764', '10.1109/LSP.2024.3456007']


def request(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'ReID-literature-research/1.0'})
    with urllib.request.urlopen(req, timeout=35) as response:
        raw = response.read()
        return raw, {'url': url, 'resolved_url': response.url, 'status': response.status,
                     'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def metadata(doi):
    try:
        raw, audit = request('https://api.openalex.org/works/https://doi.org/' + doi)
        obj = json.loads(raw)
        words = obj.get('abstract_inverted_index') or {}
        positions = {n: word for word, offsets in words.items() for n in offsets}
        audit.update(doi=doi, title=obj.get('title'), year=obj.get('publication_year'),
                     authors=[a['author']['display_name'] for a in obj.get('authorships', [])],
                     locations=obj.get('locations'),
                     abstract=' '.join(positions[n] for n in sorted(positions)),
                     scope='Metadata and indexed abstract; method claims need primary full text')
        return audit
    except Exception as exc:
        return {'doi': doi, 'error': str(exc)}


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(metadata, DOIS))
    fulltext = {}
    try:
        raw, fulltext = request('https://arxiv.org/pdf/2207.09046')
        if not raw.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        dest = ROOT.parent/'reference-paper'/'DPM_2022_arxiv_2207.09046.pdf'
        if dest.exists() and dest.read_bytes() != raw:
            raise ValueError('Different local PDF exists; do not overwrite')
        dest.write_bytes(raw)
        fulltext['local_path'] = '../reference-paper/' + dest.name
        from pypdf import PdfReader
        pdf = PdfReader(dest)
        pages = ['=== PDF PAGE %d ===\n%s' % (i + 1, page.extract_text()) for i, page in enumerate(pdf.pages)]
        textpath = ROOT/'research/paper_text/DPM_2022.txt'
        textpath.write_text('\n\n'.join(pages), encoding='utf-8')
        fulltext.update(pages=len(pdf.pages), text_sha256=hashlib.sha256(textpath.read_bytes()).hexdigest())
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(dest)
        for n in (3, 4):
            document[n-1].render(scale=1.6).to_pil().save(ROOT/'tmp/pdfs'/f'DPM_2022_{n}.png')
    except Exception as exc:
        fulltext['error'] = str(exc)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'metadata': records,
              'DPM_primary_preprint': fulltext,
              'scope': 'Focused prior art check; not an exhaustive novelty assessment. TGPG full text not obtained.'}
    (ROOT/'research/prototype_prior_art_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
