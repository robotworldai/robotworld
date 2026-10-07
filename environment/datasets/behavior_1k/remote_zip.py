"""Read selected public HF ZIP members with validated HTTP ranges; never fetch the whole archive."""
import io
import json
import re
import time
import urllib.request
from collections import OrderedDict


class RangeFile(io.RawIOBase):
    def __init__(self, url, block_size=4*1024*1024):
        self.url=url; self.pos=0; self.block_size=block_size; self.cache=OrderedDict();self.transferred=0
        data, total=self._range(0,0)
        self.size=total

    def _range(self,start,end):
        # Range in query prevents intermediaries from reusing a cached different range.
        url=self.url+('&' if '?' in self.url else '?')+f'world_range={start}-{end}'
        error=None
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if response.status!=206 or not match or int(match[1])!=start or int(match[2])!=end:
                        raise RuntimeError('Server did not honor requested byte range; refusing a full download')
                    data=response.read(end-start+2)
                    if len(data)!=end-start+1:raise RuntimeError('Truncated range')
                    self.transferred+=len(data)
                    return data,int(match[3])
            except Exception as exc:
                error=exc
                if attempt<3:time.sleep(1+attempt)
        raise error

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,size=-1):
        size=min(self.size-self.pos, size if size>=0 else self.size-self.pos)
        parts=[]
        while size>0:
            block=self.pos//self.block_size
            if block not in self.cache:
                start=block*self.block_size
                self.cache[block]=self._range(start,min(start+self.block_size,self.size)-1)[0]
                if len(self.cache)>8:self.cache.popitem(last=False)
            data=self.cache[block];self.cache.move_to_end(block)
            offset=self.pos%self.block_size;n=min(size,len(data)-offset)
            parts.append(data[offset:offset+n]);self.pos+=n;size-=n
        return b''.join(parts)


def resolve_archive(filename):
    with urllib.request.urlopen('https://huggingface.co/api/datasets/behavior-1k/zipped-datasets',timeout=30) as response:
        info=json.load(response)
    return f"https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/{info['sha']}/{filename}",info['sha']
