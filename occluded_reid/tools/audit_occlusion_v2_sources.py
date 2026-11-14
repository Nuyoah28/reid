"""Record focused prior-art leads and primary metadata without claiming exhaustive novelty."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    '"occluded person" "prototype" "mask"',
    '"occluded person" "leave-one-out"',
    '"occluded person" "reliability" "discriminative"',
    '"person re-identification" "counterfactual" "occlusion"',
]


def fetch(item):
    try:
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'ReID-research-literature-audit/1.0'})
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read()
            item.update(status=response.status, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        data = json.loads(payload)
        if item['kind'] == 'openalex_discovery':
            item['results'] = [{'title': r.get('title'), 'year': r.get('publication_year'), 'doi': r.get('doi'),
                                'landing_page': (r.get('primary_location') or {}).get('landing_page_url')}
                               for r in data.get('results', [])]
        else:
            item['metadata'] = {k: data.get('message', {}).get(k) for k in
                                ('title', 'container-title', 'published', 'published-online', 'published-print', 'DOI', 'URL')}
    except Exception as exc:
        item['error'] = str(exc)
    return item


def main():
    catalogue = list(csv.DictReader((ROOT.parent/'reference-paper/文件清单.csv').open(encoding='utf-8-sig')))
    local = []
    for row in catalogue:
        if row['编号'] not in ('R0624', 'R0369', 'R0354', 'R0353'):
            continue
        pdf = ROOT.parent/'reference-paper'/row['PDF文件名']
        text = ROOT/'research/paper_text'/f"{row['编号']}.txt"
        local.append({'id': row['编号'], 'title_listed': row['标题'], 'doi_listed': row.get('DOI'),
                      'pdf_exists': pdf.exists(), 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else None,
                      'extracted_full_text_exists': text.exists(), 'text_sha256': hashlib.sha256(text.read_bytes()).hexdigest() if text.exists() else None})
    jobs = [{'kind': 'openalex_discovery', 'query': q, 'url': 'https://api.openalex.org/works?'+urllib.parse.urlencode(
                {'search': q, 'per-page': 5, 'select': 'id,doi,title,publication_year,primary_location'})} for q in QUERIES]
    for doi in ('10.1109/TIFS.2023.3318956', '10.1145/3664647.3680958'):
        jobs.append({'kind': 'crossref_primary_metadata', 'url': 'https://api.crossref.org/works/'+doi})
    with ThreadPoolExecutor(max_workers=6) as pool:
        online = list(pool.map(fetch, jobs))
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'local_primary_documents': local,
              'online': online,
              'reviewed_scope': {'RGANet': 'Full method: RGM and RAM, especially equations 4-10, local PDF pages 4-6; discriminative FC indicator and body-part-category memory-bank invariance already exist.',
                                 'ProFD': 'Prompt guidance, noisy segmentation alignment, hybrid attention and memory-based self-distillation; do not claim prompts/visibility/distillation as new.',
                                 'SPT': 'Data-driven saliency-guided patch transfer is prior art. The project rectangle pasting does not reproduce its saliency/OIoU/mask rolling.',
                                 'DPM_2022': 'Found in SPT/ProFD bibliographies only. Full paper not yet read; prototype-mask prior art remains an open novelty risk.'},
              'claim_boundary': 'Focused audit, not exhaustive novelty search. Missing/failed web requests are recorded explicitly. No latest-SOTA or first-ever claim.'}
    destination = ROOT/'research/occlusion_v2_source_audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'local_documents': len(local), 'online': [{'kind': x['kind'], 'status': x.get('status'), 'error': x.get('error')} for x in online]}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
