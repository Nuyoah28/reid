from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote, urlsplit

root = Path(__file__).resolve().parents[2]
documents = [root/'README.md', root/'infrared_multimodal/README.md', root/'lightweight_occluded_reid/README.md',
             root/'occluded_reid/README.md']
documents += [root/'occluded_reid/research'/p for p in (
    '第二阶段实验报告.md', '第二阶段方法与来源.md', '原型方法查重记录.md',
    'architecture/第二阶段网络图.html')]
documents.append(root/'occluded_reid/results/peer_v2_summary/table.md')
links, failures = [], []
for path in documents:
    source = path.read_text(encoding='utf-8')
    targets = re.findall(r'\]\(([^)]+)\)', source) if path.suffix == '.md' else re.findall(r'(?:href|src)="([^"]+)"', source)
    for target in targets:
        url = urlsplit(target.strip('<>'))
        if url.scheme or url.netloc or not url.path:
            continue
        resolved = (path.parent/unquote(url.path)).resolve()
        row = {'document': path.relative_to(root).as_posix(), 'target': target, 'exists': resolved.exists()}
        links.append(row)
        if not row['exists']:
            failures.append(row)
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
forbidden, oversized, unexpected_images = [], [], []
for name in filter(None, tracked):
    p = root/name
    if not p.exists():
        failures.append({'staged_but_missing': name})
        continue
    if p.suffix.lower() in ('.zip', '.7z', '.rar', '.pdf', '.pt', '.pth', '.ckpt', '.onnx', '.safetensors'):
        forbidden.append(name)
    if p.stat().st_size > 10*2**20:
        oversized.append(name)
    if p.suffix.lower() in ('.jpg', '.jpeg', '.tif', '.bmp') and '/code/deep-person-reid/docs/figures/' not in name:
        unexpected_images.append(name)
report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'passed' if not (failures or forbidden or oversized or unexpected_images) else 'failed',
          'document_count': len(documents), 'local_links_checked': len(links),
          'tracked_file_count': len([x for x in tracked if x]), 'failures': failures,
          'forbidden_assets': forbidden, 'files_over_10MiB': oversized,
          'unexpected_photograph_files': unexpected_images,
          'scope': 'Current documentation local link destinations and staged file inventory; external URLs are outside this check.'}
(root/'occluded_reid/research/delivery_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
if report['status'] != 'passed':
    raise SystemExit(1)
