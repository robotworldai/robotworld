"""No model: check MCP images, isolation and idempotent native budget execution."""
import argparse
import asyncio
import json
from pathlib import Path
import urllib.error
from environment.clients.episode import EpisodeClient
from environment.bridges.mcp import call, tools

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=EpisodeClient()
    description=env.request('describe')
    first=env.request('observe')
    assert first==env.request('observe'), 'Queries changed observation'
    blocks=asyncio.run(call('observe',{}))
    image_count=sum(x.type=='image' for x in blocks)
    assert image_count>0
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/runs').exists()
    assert not Path('/world/environment/service').exists()
    assert not Path('/opt/robotworld-agent/environment/service').exists()
    assert not Path('/opt/robocasa').exists()
    request={'request_id':'harbor-budget-probe','episode_id':first['episode_id'],
        'observation_id':first['observation_id'],'commands':[
            {'tool':'set_gripper','arguments':{'note':'non-solving infrastructure probe',
                'steps':30,'targets':{'gripper_close':0}}} for _ in range(20)]}
    receipt=env.request('execute',request)
    assert env.request('execute',request)==receipt
    assert env.request('receipts/'+request['request_id'])==receipt
    assert receipt['control_step']==description['native_horizon']==450
    assert receipt['terminal']
    stale={**request,'request_id':'stale-probe'}
    try:env.request('execute',stale)
    except urllib.error.HTTPError as error:assert error.code==400
    else:raise AssertionError('Stale request was accepted')
    assert env.request('status')['control_step']==450
    (args.output/'summary.json').write_text(json.dumps({
        'classification':'infrastructure_probe_not_model_evaluation',
        'environment':env.request('status'),'duplicate_did_not_step':True,
        'query_did_not_step':True,'stale_request_rejected':True,'initial_mcp_images':image_count,
        'tools':[t.name for t in asyncio.run(tools())],
        'isolation_checks':'no simulator assets/sources/results or Docker socket at known paths',
    },indent=2))
if __name__=='__main__':main()
