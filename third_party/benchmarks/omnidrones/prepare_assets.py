"""Verify repository-provided assets without changing the pinned checkout."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def main():
    rows=json.loads((ROOT/'asset-manifest.json').read_text())
    for row in rows:
        p=ROOT/'checkout'/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Missing/corrupt original asset: '+str(p))
    print('Verified',len(rows),'repository assets; GPU/runtime not implied.')
if __name__=='__main__':main()
