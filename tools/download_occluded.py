"""Obtain the authors' public academic-use Occluded-REID archive, not DukeMTMC."""
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    # Pin both provenance and content: an updated master must not silently change
    # an experiment's benchmark or overwrite its asset manifest.
    revision='61e54da27fc1999dd68ce5681f251220beb0fc64'
    expected='1c8854e02db775da6a51399df4b7e02c299804850dff9985451d610fc8c6d57d'
    url=f'https://raw.githubusercontent.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/{revision}/Occluded_REID.zip'
    dest=ROOT/'data'/'Occluded_REID.zip';dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url,timeout=60) as response,dest.with_suffix('.part').open('wb') as output:
            done=0;last=time.monotonic()
            while chunk:=response.read(256*1024):
                output.write(chunk);done+=len(chunk)
                if time.monotonic()-last>5:print(f'Occluded-REID {done/1e6:.1f}/41.9 MB',flush=True);last=time.monotonic()
        dest.with_suffix('.part').replace(dest)
    actual=hashlib.sha256(dest.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError(f'SHA256 mismatch: {actual}')
    with zipfile.ZipFile(dest) as archive:
        if archive.testzip() is not None:raise RuntimeError('Corrupt ZIP')
        root=dest.parent.resolve()
        for item in archive.infolist():
            if not (root/item.filename).resolve().is_relative_to(root):raise RuntimeError('Unsafe ZIP')
        archive.extractall(root)
    manifest={'url':url,'sha256':actual,'upstream_revision':revision,'size':dest.stat().st_size,'use':'academic/educational only, as stated by dataset authors','protocol':'train on Market-1501; all 200 Occluded-REID identities held out; 1000 occluded queries against 1000 whole-body gallery images'}
    (ROOT/'research'/'asset_occluded.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
