"""Native RoboLab infrastructure probe; does not attempt to solve the task."""
import json
from pathlib import Path
import shlex

if __name__ != '__main__':
    from agents.robotworld import RobotWorldProbeAgent, REMOTE
    
    
    class RoboLabProbeAgent(RobotWorldProbeAgent):
        @staticmethod
        def name(): return 'robotworld-robolab-probe'
    
        async def install(self, environment):
            await super().install(environment)
            await environment.upload_file(Path(__file__), REMOTE+'/robolab_probe.py')
    
        async def run(self, instruction, environment, context):
            logs = str(self.environment_logs_dir)
            await self.exec_as_agent(environment,
                command='python '+REMOTE+'/robolab_probe.py '+shlex.quote(logs+'/robotworld'),
                env={'PYTHONPATH':REMOTE},timeout_sec=900)
            context.metadata={'classification':'ten_step_infrastructure_probe_not_model_evaluation'}
    

def probe(output):
    import asyncio
    from environment.clients.episode import EpisodeClient
    from environment.bridges.mcp import call
    client = EpisodeClient()
    spec = client.request('describe')
    first = client.request('observe')
    assert spec['benchmark'] == 'robolab'
    assert first == client.request('observe')
    images = sum(block.type=='image' for block in asyncio.run(call('observe',{})))
    assert images == 2
    for hidden in ('/var/run/docker.sock', '/runs', '/opt/robolab', '/world/environment/service'):
        assert not Path(hidden).exists(), hidden
    request={'request_id':'native-hold-10','episode_id':first['episode_id'],
             'observation_id':first['observation_id'],'commands':[{
                 'tool':'set_gripper','arguments':{'note':'hold current measured arm, open gripper',
                                                 'steps':10,'targets':{'gripper_close':0}}}]}
    receipt=client.request('execute',request)
    assert receipt == client.request('execute',request)
    assert receipt == client.request('receipts/native-hold-10')
    assert 0 < receipt['control_step'] <= 10
    assert receipt['termination'] != 'infrastructure_error'
    after = client.request('observe')
    assert after == client.request('observe')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps({
        'classification':'ten_step_infrastructure_probe_not_model_evaluation',
        'description':spec,'environment':receipt,'initial_mcp_images':images,
        'duplicate_did_not_step':True,'query_did_not_step':True,
        'initial_state':first['observation']['state'],'final_state':after['observation']['state'],
        'final_image_steps':[i['control_step'] for i in after['observation']['images']],
        'termination_note':'Probe stops voluntarily; native task horizon is unchanged.'},indent=2))


if __name__=='__main__':
    import sys
    probe(sys.argv[1])
