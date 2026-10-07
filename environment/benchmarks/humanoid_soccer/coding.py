"""Bounded feedback-program RPC, separate from simulator process/state."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys


class ControllerProgram:
    def __init__(self,code):
        self.process=subprocess.Popen([sys.executable,'-I','-S',str(Path(__file__).with_name('program_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            env={},cwd='/tmp',start_new_session=True)
        self.buffer=b''
        try:self.request({'code':code})
        except BaseException:self.close();raise

    def request(self,data):
        payload=json.dumps(data,allow_nan=False).encode()+b'\n'
        if len(payload)>262144:raise ValueError('Controller input too large')
        self.process.stdin.write(payload);self.process.stdin.flush()
        import time
        deadline=time.monotonic()+2
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                raise ValueError('Control program exceeded 2s callback deadline')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:raise ValueError('Control worker terminated (check code/resource limits)')
            self.buffer+=chunk
            if len(self.buffer)>131072:raise ValueError('Control worker output too large')
        line,self.buffer=self.buffer.split(b'\n',1)
        result=json.loads(line)
        if 'error' in result:raise ValueError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:self.process.kill()
        self.process.wait()
        self.process.stdin.close();self.process.stdout.close()
