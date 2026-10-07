"""Share identical read-only scene resources by hardlink; preserve paths and bytes."""
import argparse,hashlib,json,os
from collections import defaultdict
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args();groups=defaultdict(list)
 for manifest in sorted((WORLD/'Assets/robodojo/scenes').glob('*/asset-manifest.json')):
  data=json.loads(manifest.read_text())
  for row in data['files']:groups[(row['bytes'],row['sha256'])].append(manifest.parent/'Assets'/row['path'])
 records=[];saved=0
 for (size,digest),paths in groups.items():
  paths=list(dict.fromkeys(paths))
  if len(paths)<2:continue
  original=paths[0]
  if sha(original)!=digest:raise ValueError('Source checksum mismatch: '+str(original))
  for target in paths[1:]:
   if os.path.samefile(original,target):continue
   if sha(target)!=digest:raise ValueError('Target checksum mismatch: '+str(target))
   if a.apply:
    tmp=target.with_name(target.name+'.world-link-tmp')
    if tmp.exists():raise FileExistsError(tmp)
    os.link(original,tmp)
    try:os.replace(tmp,target)
    finally:
     if tmp.exists():tmp.unlink()
   saved+=size;records.append({'source':str(original.relative_to(WORLD)),'target':str(target.relative_to(WORLD)),'bytes':size,'sha256':digest})
 out=WORLD/'reports/assets/robodojo-deduplication.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'applied':a.apply,'bytes_shared':saved,'identical_files':records,'asset_contents_changed':False},indent=2));print(json.dumps({'applied':a.apply,'files':len(records),'bytes_shared':saved}))
if __name__=='__main__':main()
