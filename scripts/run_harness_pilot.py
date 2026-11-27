"""Start a fresh native CloseDrawer episode, then connect a neutral harness."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from environment.clients.episode import EpisodeClient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness',choices=['probe','claude-code','codex'],default='probe')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=ROOT/'Assets/robocasa/data')
    p.add_argument('--port',type=int,default=18765)
    p.add_argument('--claude',default='claude');p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--wall-timeout',type=int,default=43200)
    a=p.parse_args()
    if a.wall_timeout<=0:p.error('--wall-timeout must be positive')
    if a.harness=='claude-code' and not all(os.environ.get(k) for k in ('ANTHROPIC_BASE_URL','ANTHROPIC_API_KEY')):
        p.error('Load ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY first')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sim=out/'simulation';sim.mkdir()
    env=os.environ.copy();env['WORLD_EPISODE_TOKEN']=secrets.token_hex(32)
    env['WORLD_EPISODE_URL']=f'http://127.0.0.1:{a.port}'
    env['PYTHONPATH']=str(ROOT)
    name='world-episode-'+secrets.token_hex(6)
    (out/'launch.json').write_text(json.dumps({'code_root':str(ROOT),'harness':a.harness,'model':a.model,'wall_timeout':a.wall_timeout,'environment_mount':str(ROOT/'environment'),'assets':str(a.assets.resolve()),'python':sys.executable,'codex_source':os.environ.get('WORLD_CODEX_SOURCE',str(ROOT/'codex')) if a.harness=='codex' else None},indent=2))
    command=['docker','run','-d','--name',name,'--gpus','all','--shm-size','2g',
        '-e','WORLD_EPISODE_TOKEN','-p',f'127.0.0.1:{a.port}:8765',
        '-v',str(ROOT/'environment')+':/world/environment:ro',
        '-v',str(a.assets.resolve())+':/opt/robocasa/robocasa/models/assets',
        '-v',str(sim)+':/runs','world/robocasa:1.0.1','python','-m',
        'environment.service.server','--host','0.0.0.0','--output','/runs']
    subprocess.run(command,env=env,check=True,stdout=subprocess.DEVNULL)
    client=EpisodeClient(env['WORLD_EPISODE_URL'],env['WORLD_EPISODE_TOKEN'])
    try:
        deadline=time.monotonic()+300
        while True:
            try:s=client.request('status');break
            except (urllib.error.URLError,ConnectionError):
                if time.monotonic()>deadline:raise TimeoutError('Simulator did not start')
                time.sleep(1)
        if a.harness=='probe':
            # Execute a non-solving neutral action to the original episode budget.
            # This verifies accounting, not reachability or model performance.
            first=client.request('observe');assert first==client.request('observe')
            request={'request_id':'budget-probe','episode_id':s['episode_id'],
                'observation_id':s['observation_id'],'commands':[
                    {'tool':'set_gripper','arguments':{'note':'non-solving protocol verification',
                     'steps':30,'targets':{'gripper_close':0}}} for _ in range(20)]}
            r=client.request('execute',request)
            assert client.request('execute',request)==r
            assert r['terminal'] and r['control_step']<=s['horizon']
            (out/'probe.json').write_text(json.dumps({'classification':'infrastructure_probe_not_model_evaluation',
                'receipt':r,'duplicate_did_not_step':True,'queries_did_not_step':True},indent=2))
        elif a.harness=='codex':
            subprocess.run([sys.executable,str(ROOT/'harnesses/codex/run.py'),
                '--model',a.model,'--output',str(out/'harness'),'--wall-timeout',str(a.wall_timeout)],env=env,check=True)
        else:
            subprocess.run([sys.executable,str(ROOT/'harnesses/claude_code/run.py'),
                '--claude',a.claude,'--model',a.model,'--output',str(out/'harness'),'--wall-timeout',str(a.wall_timeout)],env=env,check=True)
    finally:
        subprocess.run(['docker','stop','--time','30',name],stdout=subprocess.DEVNULL,check=False)
        with (out/'simulator.log').open('w') as f:
            subprocess.run(['docker','logs',name],stdout=f,stderr=subprocess.STDOUT,check=False)
        subprocess.run(['docker','rm',name],stdout=subprocess.DEVNULL,check=False)
    print(out)

if __name__=='__main__':main()
