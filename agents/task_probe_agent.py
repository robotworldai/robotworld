"""Short non-solving Harbor transport probe for AI-CPS/WheeledLab/BEHAVIOR."""
import argparse
import json
from pathlib import Path


def run_probe(output):
    from environment.clients.episode import EpisodeClient
    client=EpisodeClient()
    description=client.request('describe')
    first=client.request('observe')
    assert first==client.request('observe'), 'Read-only query advanced/changed observation'
    names={tool['name'] for tool in description['tools']}
    if 'drive' in names:
        command={'tool':'drive','arguments':{'note':'Non-solving transport probe','steps':3,'action':[0,0]}}
    elif 'move_joints' in names:
        command={'tool':'move_joints','arguments':{'note':'Non-solving transport probe','steps':3,'arm_action':[0]*7}}
    elif 'set_grippers' in names:
        command={'tool':'set_grippers','arguments':{'note':'Non-solving transport probe','steps':3,'targets':{'left_gripper':1,'right_gripper':1}}}
    else:
        raise ValueError('No reviewed neutral action for this task; add an explicit probe profile')
    request=dict(request_id='harbor-transport-probe',episode_id=first['episode_id'],observation_id=first['observation_id'],commands=[command])
    receipt=client.request('execute',request)
    assert receipt['executed_steps']>0 and not receipt['execution_uncertain']
    assert client.request('execute',request)==receipt
    assert client.request('receipts/'+request['request_id'])==receipt
    last=client.request('observe')
    assert last['control_step']==receipt['control_step']
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(dict(classification='infrastructure_probe_not_model_evaluation',
        description=description,receipt=receipt,images=len(last['observation']['images']),query_did_not_step=True,retry_did_not_step=True),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run_probe(parser.parse_args().output)
else:
    import shlex
    from agents.robotworld import RobotWorldProbeAgent,ROOT,REMOTE
    class RobotWorldTaskProbeAgent(RobotWorldProbeAgent):
        @staticmethod
        def name():return 'robotworld-task-probe'
        async def install(self,environment):
            await super().install(environment)
            await environment.upload_file(ROOT/'agents/task_probe_agent.py',REMOTE+'/agents/task_probe_agent.py')
        async def run(self,instruction,environment,context):
            await self.exec_as_agent(environment,command='python -m agents.task_probe_agent --output '+shlex.quote(str(self.environment_logs_dir)+'/robotworld'),env={'PYTHONPATH':REMOTE},timeout_sec=3600)
            context.metadata={'classification':'infrastructure_probe_not_model_evaluation','execution':'inside-agent-container'}
