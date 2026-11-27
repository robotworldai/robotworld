"""Finalization signals must never interrupt simulator mutation/receipt commit."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest


@pytest.mark.parametrize('stop_signal', ['SIGUSR1', 'SIGTERM'])
def test_signal_waits_for_native_step_and_receipt(tmp_path, stop_signal):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    script = '''
import os, signal, time
from pathlib import Path
from environment.service.http import serve_episode
root=Path(os.environ['PROBE_OUTPUT'])
class Adapter:
 def describe(self): return {'tools':[]}
 def observe(self): return {}
 def prepare(self, commands): return [(0,1)]
 def step(self, action):
  (root/'order').write_text('step_start\\n')
  os.kill(os.getpid(), getattr(signal, os.environ['PROBE_SIGNAL']))
  time.sleep(.1)
  with (root/'order').open('a') as f: f.write('step_end\\n')
  return dict(success=False, terminated=False, truncated=False)
 def close(self):
  assert 'execution_completed' in (root/'events.jsonl').read_text()
  with (root/'order').open('a') as f: f.write('close\\n')
serve_episode(Adapter(),10,root,host='127.0.0.1',port=int(os.environ['PROBE_PORT']))
'''
    env = {**os.environ, 'WORLD_EPISODE_TOKEN':'test-token-only-0123456789',
           'PROBE_OUTPUT':str(tmp_path), 'PROBE_PORT':str(port), 'PROBE_SIGNAL':stop_signal}
    process = subprocess.Popen([sys.executable, '-c', script], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    headers = {'Authorization':'Bearer '+env['WORLD_EPISODE_TOKEN'],
               'Content-Type':'application/json'}
    def request(path, data=None):
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+path,
                                     data=None if data is None else json.dumps(data).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'episode.json').exists():
            if process.poll() is not None or time.monotonic()>deadline:
                pytest.fail('Service failed to start: '+process.stderr.read().decode())
            time.sleep(.02)
        state=request('status')
        result=request('execute', {'episode_id':state['episode_id'],
            'observation_id':state['observation_id'], 'request_id':'probe', 'commands':[1]})
        assert result['executed_steps']==1
        while not (tmp_path/'finalized').exists() and time.monotonic()<deadline:
            time.sleep(.02)
        assert (tmp_path/'finalized').read_text().strip()=='complete'
        assert (tmp_path/'order').read_text().splitlines()==['step_start','step_end','close']
        state=json.loads((tmp_path/'episode.json').read_text())
        assert state['control_step']==1 and state['termination']=='agent_stopped'
    finally:
        if process.poll() is None: process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill();process.wait()
