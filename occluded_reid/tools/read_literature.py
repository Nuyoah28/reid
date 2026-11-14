"""Index supplied PDFs without treating filenames as verified publication metadata."""
import csv
import json
import re
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT.parent / 'reference-paper'
DEST = ROOT / 'research' / 'paper_text'
SELECTED = {'R0032','R0038','R0052','R0118','R0150','R0151','R0186','R0294','R0295','R0299','R0320','R0353','R0354','R0356','R0369','R0431','R0454','R0462','R0484','R0513','R0574','R0576','R0593','R0594','R0624'}

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = list(csv.DictReader((REFERENCES/'文件清单.csv').open(encoding='utf-8-sig')))
    index=[]
    for row in records:
        path=REFERENCES/row['PDF文件名']
        if not path.is_file():
            continue
        target=DEST/(row['编号']+'.txt')
        try:
            reader=PdfReader(path)
            pages=reader.pages if row['编号'] in SELECTED else reader.pages[:2]
            expected_pages=len(pages)
            cached=target.read_text(encoding='utf-8') if target.exists() else ''
            if cached.count('--- PAGE ')!=expected_pages:
                # Some PDFs expose isolated UTF-16 surrogate codepoints. Preserve
                # readable text and replace only these malformed characters.
                target.write_text('\n\n'.join(f'--- PAGE {i+1} ---\n{p.extract_text()}' for i,p in enumerate(pages)), encoding='utf-8',errors='replace')
            body=target.read_text(encoding='utf-8')
            index.append({'id':row['编号'],'title_listed':row['标题'],'year_listed':row['年份'],'venue_listed':row['来源'],'ccf_listed':row['CCF2026'],'pdf':(Path('..')/'reference-paper'/row['PDF文件名']).as_posix(), 'pages':len(reader.pages),'extraction_scope':'full_text' if row['编号'] in SELECTED else 'first_2_pages','links':sorted({url.rstrip('.,);') for url in re.findall(r'https?://[^\s<>]+',body)})})
            if len(index)%25==0: print(f'Indexed {len(index)} PDFs', flush=True)
        except Exception as exc:
            index.append({'id':row['编号'],'error':str(exc)})
    (ROOT/'research'/'literature_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Finished {len(index)} available PDFs / {len(records)} catalogue entries',flush=True)

if __name__=='__main__':
    main()
