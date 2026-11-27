"""Harbor installed macro-action transport diagnostic."""
import shlex
from agents.robotworld import RobotWorldProbeAgent,ROOT,REMOTE

class RobotWorldDojoProbeAgent(RobotWorldProbeAgent):
    @staticmethod
    def name():return 'robotworld-robodojo-probe'
    async def install(self,environment):
        await super().install(environment)
        await environment.upload_file(ROOT/'agents/robodojo_probe.py',REMOTE+'/agents/robodojo_probe.py')
    async def run(self,instruction,environment,context):
        await self.exec_as_agent(environment,command='python -m agents.robodojo_probe --output '+shlex.quote(str(self.environment_logs_dir)+'/robotworld'),env={'PYTHONPATH':REMOTE},timeout_sec=3600)
        context.metadata={'classification':'infrastructure_probe_not_model_evaluation','execution':'inside-agent-container'}
