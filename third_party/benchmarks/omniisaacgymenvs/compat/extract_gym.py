#!/usr/bin/env python3
"""Extract the unchanged legacy runtime extension from a locally licensed official image."""
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
import uuid
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--image',default='nvcr.io/nvidia/isaac-sim:4.0.0@sha256:aad570146698a4c1c9e9b834638ddccfc0675af110657203e369625985c85c66')
a=p.parse_args()
manifest=json.loads((ROOT/'vendor-provenance.json').read_text())
target=ROOT/'vendor/omni.isaac.gym'
if target.exists():
    if all((ROOT/x['path']).is_file() and hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in manifest['files']):
        print('Original extension already present; manifest verified.')
        raise SystemExit(0)
    raise SystemExit('Existing vendor differs from manifest; inspect manually, refusing overwrite.')
name='world-t14-extract-'+uuid.uuid4().hex[:12]
(ROOT/'vendor').mkdir(exist_ok=True)
subprocess.run(['docker','create','--name',name,a.image],check=True)
try:
    subprocess.run(['docker','cp',name+':/isaac-sim/exts/omni.isaac.gym',str(ROOT/'vendor')],check=True)
finally:
    subprocess.run(['docker','rm',name],check=True)
for row in manifest['files']:
    if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit('Extracted extension differs from pinned runtime manifest: '+row['path'])
print('Verified original extension. Keep local; retained proprietary license governs use.')
