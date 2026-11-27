"""Exercise one original move_eef macro using only public observed hand state."""
import argparse,json,re
from pathlib import Path
from environment.clients.episode import EpisodeClient


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    env=EpisodeClient(); description=env.request('describe');observation=env.request('observe')
    assert description['benchmark']=='robodojo'
    assert observation==env.request('observe')
    text='\n'.join(observation['observation']['feedback_text'])
    keys=('left_x','left_y','left_z','left_pitch_deg','left_roll_deg','left_yaw_deg','left_gripper')
    state={key:float(re.findall(r'^'+key+r'=([-+0-9.eE]+)$',text,re.M)[-1]) for key in keys}
    receipts=[]
    for index in range(2):
        current=env.request('status')
        if current['terminal']:break
        targets=dict(state)
        if index:targets['left_z']+=0.002
        request={'request_id':f'dojo-observed-hold-{index}','episode_id':current['episode_id'],
                 'observation_id':current['observation_id'],'commands':[{'tool':'move_eef','arguments':{
                     'targets':targets,'note':'Infrastructure probe: hold the observed grasp pose; if no physical steps, lift 2 mm with unchanged orientation and gripper.'}}]}
        receipt=env.request('execute',request)
        assert not receipt['execution_uncertain']
        assert env.request('execute',request)==receipt
        assert env.request('receipts/'+request['request_id'])==receipt
        receipts.append(receipt)
        if receipt['executed_steps']>0:break
    assert any(r['executed_steps']>0 for r in receipts),'Neither observed hold nor 2 mm lift advanced physics'
    (a.output/'summary.json').write_text(json.dumps({'classification':'infrastructure_probe_not_model_evaluation',
        'public_observed_targets':state,'receipts':receipts,'duplicate_did_not_step':True,'query_did_not_step':True},indent=2))

if __name__=='__main__':main()
