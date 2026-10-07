"""Extract selected original members from a pinned official ZIP using HTTP ranges."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from .remote_zip import RangeFile


def fetch(index_path, destination, selected):
    index=json.loads(Path(index_path).read_text());destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    raw=RangeFile(index['url']);records=[]
    with zipfile.ZipFile(raw) as archive:
        for name in sorted(set(selected)):
            info=archive.getinfo(name)
            if info.is_dir():continue
            target=(destination/name).resolve()
            target.relative_to(destination.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                temporary=target.with_suffix(target.suffix+'.part')
                with archive.open(name) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                temporary.replace(target)
            # CRC check also validates cached originals on rerun.
            import zlib
            data=target.read_bytes()
            if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError(f'Asset integrity mismatch: {name}')
            records.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            if len(records)%25==0:print('verified',len(records),'members',flush=True)
    return {'url':index['url'],'revision':index['revision'],'range_download_bytes':raw.transferred,'files':records}
