"""Download only selected native asset directories from the official HF dataset."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]/'third_party/benchmarks/bench2dex'
REPO = 'Bench2Dex/Assets'


def request_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response), response.headers.get('Link', '')
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def inventory():
    existing=ROOT/'asset-manifest.json'
    old=json.loads(existing.read_text()) if existing.exists() else None
    if old:rev=old['revision']
    else:
        info,_=request_json(f'https://huggingface.co/api/datasets/{REPO}');rev=info['sha']
    found={x['path']:x for x in old['files']} if old else {}
    prefixes=json.loads((ROOT/'asset-prefixes.json').read_text())
    def scan(prefix):
        entries_found=[]
        url = f'https://huggingface.co/api/datasets/{REPO}/tree/{rev}/'+urllib.parse.quote(prefix, safe='/')+'?recursive=true&limit=1000'
        while url:
            entries, link = request_json(url)
            for item in entries:
                if item['type'] == 'file':
                    entries_found.append(item)
            url = None
            for entry in link.split(','):
                if 'rel="next"' in entry:
                    url = entry.split('<',1)[1].split('>',1)[0]
        print('Indexed',prefix,flush=True)
        return entries_found
    pending=[p for p in prefixes if not any(x.startswith(p+'/') for x in found)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for entries in pool.map(scan,pending):
            found.update({x['path']:x for x in entries})
    manifest = {'repository':REPO,'revision':rev,'prefixes':prefixes,'files':list(found.values())}
    (ROOT/'asset-manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def download(item, revision):
    rel = Path(item['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe repository path')
    target = ROOT/'dex2bench_dataset'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    expected = item.get('lfs',{}).get('oid')
    def valid():
        if not target.exists() or target.stat().st_size != item['size']:
            return False
        return not expected or hashlib.sha256(target.read_bytes()).hexdigest() == expected
    if valid():
        return
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'+urllib.parse.quote(str(rel),safe='/')+'?download=true'
    temp = target.with_name(target.name+'.partial')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=120) as response, temp.open('wb') as stream:
                while data := response.read(1024*1024):
                    stream.write(data)
            temp.replace(target)
            if not valid():
                raise ValueError('Asset size or SHA256 mismatch: '+str(rel))
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--inventory-only',action='store_true');a=p.parse_args()
    path=ROOT/'asset-manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else inventory()
    print('Files:',len(manifest['files']),'bytes:',sum(x['size'] for x in manifest['files']),flush=True)
    if a.inventory_only:
        return
    ground_manifest=json.loads((ROOT/'shared-assets.json').read_text())
    for entry in ground_manifest['files']:
        dest=ROOT/entry['path']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==entry['sha256']:continue
        with urllib.request.urlopen(entry['url'],timeout=120) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Ground asset hash mismatch')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,x,manifest['revision']) for x in manifest['files']]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if i%25==0:print('Downloaded/verified',i,flush=True)
    print('Assets complete',flush=True)


if __name__=='__main__':
    main()
