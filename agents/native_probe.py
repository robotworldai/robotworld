"""Non-solving native action/receipt transport check; never writes reward artifacts."""
import argparse
import json
from pathlib import Path
from environment.clients.episode import EpisodeClient


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--complete-episode',action='store_true'); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    env=EpisodeClient(); description=env.request('describe'); first=env.request('observe')
    assert first==env.request('observe')
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/runs').exists()
    assert not Path('/opt/robotworld/environment/service').exists()
    meta=description['action_metadata']
    action=[max(lo if lo is not None else float('-inf'),min(hi if hi is not None else float('inf'),0.)) for lo,hi in zip(meta['lower'],meta['upper'])]
    request={'request_id':'native-infrastructure-probe','episode_id':first['episode_id'],
             'observation_id':first['observation_id'],'commands':[{'tool':'apply_action','arguments':{
                 'action':action,'steps':min(10,description['native_horizon']),'note':'Non-solving transport probe'}}]}
    receipt=env.request('execute',request)
    assert receipt['executed_steps']>0
    assert not receipt['execution_uncertain']
    assert env.request('execute',request)==receipt
    assert env.request('receipts/'+request['request_id'])==receipt
    receipts=[receipt]
    if args.complete_episode:
        while not receipt['terminal']:
            state=env.request('status')
            request={'request_id':'native-budget-'+str(state['control_step']), 'episode_id':state['episode_id'],
                     'observation_id':state['observation_id'],'commands':[{'tool':'apply_action','arguments':{
                         'action':action,'steps':min(50,state['horizon']-state['control_step']),'note':'Non-solving native budget probe'}}]}
            receipt=env.request('execute',request)
            assert receipt['executed_steps']>0 and not receipt['execution_uncertain']
            receipts.append(receipt)
    (args.output/'summary.json').write_text(json.dumps({'classification':'infrastructure_probe_not_model_evaluation',
        'description':description,'initial_step':first['control_step'],'receipt':receipt,'receipts':receipts,
        'query_did_not_step':True,'duplicate_did_not_step':True},indent=2))

if __name__=='__main__':main()
