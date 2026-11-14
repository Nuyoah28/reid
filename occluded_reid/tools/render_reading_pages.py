"""Render selected method pages for visual reading with bundled pypdfium2."""
from pathlib import Path
import json
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
index=json.loads((ROOT/'research'/'literature_index.json').read_text(encoding='utf-8'))
sources={r['id']:(ROOT/r['pdf'] if (ROOT/r['pdf']).exists() else ROOT.parent/r['pdf']) for r in index if 'pdf' in r}
sources.update({'OSNet':ROOT/'tmp/sources/osnet_paper.pdf','CAL':ROOT/'tmp/sources/cal_paper.pdf'})
dest=ROOT/'tmp'/'pdfs';dest.mkdir(parents=True,exist_ok=True)
for key,page_number in [('OSNet',3),('CAL',3),('R0353',3),('R0052',3)]:
    pdf=pdfium.PdfDocument(sources[key])
    page=pdf[page_number-1]
    page.render(scale=1.5).to_pil().save(dest/f'{key}_{page_number}.png')
