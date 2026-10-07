"""Verify official, locally copied assets; retrieve missing LFS files only if needed."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
source=root/'checkout'
files=[p for folder in ('ckp','motions/soccer-standard','source/whole_body_tracking/soccer/assets/unitree_description') for p in (source/folder).rglob('*') if p.is_file()]
required=[source/'ckp/policy_30000.onnx',source/'source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1_actuator.xml']
if not files or any(not p.is_file() for p in required):raise SystemExit('Restore official checkout first: python scripts/fetch_sources.py humanoid_soccer')
pointers=[p for p in files if p.stat().st_size<1024 and p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/')]
if pointers:
    subprocess.run(['git','-C',str(source),'lfs','pull','--include=ckp/**,motions/soccer-standard/**,source/whole_body_tracking/soccer/assets/unitree_description/**'],check=True)
    if any(p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/') for p in pointers):raise SystemExit('Unresolved LFS pointers')
manifest=[{'path':str(p.relative_to(source)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
(root/'assets.local.json').write_text(json.dumps(manifest,indent=2))
print(f'Verified {len(files)} official files; {sum(x["bytes"] for x in manifest)} bytes. No benchmark-wide asset download.')
