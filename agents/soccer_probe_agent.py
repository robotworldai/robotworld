"""Full native-horizon neutral-target probe, not a soccer reference solution."""
import argparse
import json
from pathlib import Path


def probe(output):
    from environment.clients.episode import EpisodeClient
    client=EpisodeClient();spec=client.request('describe');first=client.request('observe')
    assert spec['benchmark']=='humanoid_soccer'
    assert first==client.request('observe')
    state=first['observation']['state'];key=next(iter(state['joint_positions']))
    commands=[];remaining=first['horizon']
    while remaining:
        n=min(50,remaining);remaining-=n
        commands.append({'tool':'move_joints','arguments':{'note':'non-solving native-horizon probe',
                         'steps':n,'joint_positions':{key:state['joint_positions'][key]}}})
    request={'request_id':'native-horizon-probe','episode_id':first['episode_id'],
             'observation_id':first['observation_id'],'commands':commands}
    receipt=client.request('execute',request)
    assert not receipt['execution_uncertain']
    assert receipt['control_step']==first['horizon'] and receipt['terminal']
    assert client.request('execute',request)==receipt
    assert client.request('receipts/native-horizon-probe')==receipt
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps({'classification':'infrastructure_probe_not_model_evaluation',
        'description':spec,'receipt':receipt,'query_did_not_step':True,'duplicate_did_not_step':True},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);probe(p.parse_args().output)
else:
    import shlex
    from agents.robotworld import RobotWorldProbeAgent, ROOT, REMOTE
    class SoccerProbeAgent(RobotWorldProbeAgent):
        @staticmethod
        def name():return 'robotworld-soccer-probe'
        async def install(self,environment):
            await super().install(environment)
            await environment.upload_file(ROOT/'agents/soccer_probe_agent.py',REMOTE+'/agents/soccer_probe_agent.py')
        async def run(self,instruction,environment,context):
            await self.exec_as_agent(environment,command='python -m agents.soccer_probe_agent --output '+shlex.quote(str(self.environment_logs_dir)+'/robotworld'),env={'PYTHONPATH':REMOTE},timeout_sec=3600)
            context.metadata={'classification':'infrastructure_probe_not_model_evaluation'}
