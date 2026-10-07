"""Fetch one original official anchor per task; never expose demonstrations to policy."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from .assets import ROOT,request_json


def main():
    repo='Bench2Dex/teleopdata';existing=ROOT/'anchor-manifest.json'
    if existing.exists():rev=json.loads(existing.read_text())['revision']
    else:
        info,_=request_json(f'https://huggingface.co/api/datasets/{repo}');rev=info['sha']
    tasks=json.loads((ROOT/'tasks.json').read_text());folder=ROOT/'anchors';folder.mkdir(exist_ok=True)
    def fetch(item):
        wid,task=item
        directory='dataset/'+task['id']+'/origin-generalization'
        entries,_=request_json(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{directory}')
        entry=next(x for x in entries if x['path']==directory+'/episode_000000.hdf5')
        dest=folder/(wid+'.hdf5')
        expected=entry.get('lfs',{}).get('oid')
        if not dest.exists() or dest.stat().st_size!=entry['size'] or (expected and hashlib.sha256(dest.read_bytes()).hexdigest()!=expected):
            url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'+entry['path']+'?download=true'
            with urllib.request.urlopen(url,timeout=120) as r: data=r.read()
            if len(data)!=entry['size'] or (expected and hashlib.sha256(data).hexdigest()!=expected):raise ValueError('Anchor hash mismatch')
            dest.write_bytes(data)
        print('Official anchor ready:',wid,flush=True)
        return {'case':wid,'path':str(dest.relative_to(ROOT)),**entry}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(fetch,tasks.items()))
    (ROOT/'anchor-manifest.json').write_text(json.dumps({'repository':repo,'revision':rev,'episode':'000000','files':files},indent=2))


if __name__=='__main__':main()
