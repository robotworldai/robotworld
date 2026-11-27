#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
from html.parser import HTMLParser
root=Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text());d=json.loads((root/'tasks.json').read_text())
assert len(d['tasks'])==17 and {t['id'] for t in d['tasks']}=={f'T{i:02d}' for i in range(1,18)}
for f in m['files']:
 p=root/f['path'];assert p.is_file(),f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],f['path']
class Links(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.links=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='a':self.links.append(a.get('href',''))
l=Links();l.feed((root/'index.html').read_text())
assert len(l.ids)==len(set(l.ids))
for href in l.links:
 if href.startswith('#'):assert href[1:] in l.ids,href
 elif href and not href.startswith(('https://','http://')):assert (root/href).is_file(),href
for t in d['tasks']:
 assert t['assets'] and t['source_snapshots'],t['id']
 for p in t['source_snapshots']:assert(root/p).is_file(),p
 for a in t['assets']:
  if a.get('evidence_package_path'):assert(root/a['evidence_package_path']).is_file()
print(json.dumps({'ok':True,'tasks':17,'files':len(m['files']),'checks':['SHA-256','task IDs','source paths','HTML relative links']},ensure_ascii=False))
