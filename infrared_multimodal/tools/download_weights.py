"""Resume HTTP ranges from the official PyTorch ImageNet URL; verify its hash."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request
import subprocess

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
SIZE, CHUNK = 102502400, 512*1024


def main():
    cache = ROOT/'tmp/weight_chunks'; cache.mkdir(parents=True, exist_ok=True)
    def download(i):
        lo, hi = i*CHUNK, min(SIZE, (i+1)*CHUNK)-1
        path = cache/f'{i:04d}.bin'
        if path.exists() and path.stat().st_size == hi-lo+1:
            return
        for attempt in range(4):
            try:
                partial, headers = path.with_suffix('.part'), path.with_suffix('.headers')
                subprocess.run(['curl.exe', '-f', '-sS', '-L', '--connect-timeout', '10',
                                '--max-time', '45', '-r', f'{lo}-{hi}', URL+'?download=1',
                                '-o', str(partial), '-D', str(headers)], check=True)
                if f'content-range: bytes {lo}-{hi}/{SIZE}' not in headers.read_text().lower():
                    raise ValueError('Unexpected range response.')
                data = partial.read_bytes()
                if len(data) != hi-lo+1:
                    raise ValueError('Incomplete chunk.')
                path.write_bytes(data)
                return
            except Exception:
                if attempt == 3:
                    raise
    total, done = (SIZE+CHUNK-1)//CHUNK, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download, i) for i in range(total)]):
            future.result(); done += 1
            if done % 20 == 0 or done == total:
                print(f'ImageNet initializer: {done}/{total} chunks', flush=True)
    output = ROOT/'weights/resnet50-19c8e357.pth'
    with output.with_suffix('.download').open('wb') as handle:
        for i in range(total):
            handle.write((cache/f'{i:04d}.bin').read_bytes())
    checksum = hashlib.sha256(output.with_suffix('.download').read_bytes()).hexdigest()
    if not checksum.startswith('19c8e357'):
        raise ValueError(checksum)
    output.with_suffix('.download').replace(output)
    print('Verified SHA256:', checksum, flush=True)


if __name__ == '__main__':
    main()
