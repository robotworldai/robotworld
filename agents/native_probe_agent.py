"""Harbor installed native-project transport probe."""
import shlex
from agents.robotworld import RobotWorldProbeAgent, ROOT, REMOTE

class RobotWorldNativeProbeAgent(RobotWorldProbeAgent):
    @staticmethod
    def name(): return 'robotworld-native-probe'

    async def install(self, environment):
        await super().install(environment)
        await environment.upload_file(ROOT/'agents/native_probe.py',REMOTE+'/agents/native_probe.py')

    async def run(self,instruction,environment,context):
        logs=str(self.environment_logs_dir)
        await self.exec_as_agent(environment,command='python -m agents.native_probe --output '+shlex.quote(logs+'/robotworld'),
                                env={'PYTHONPATH':REMOTE},timeout_sec=3600)
        context.metadata={'classification':'infrastructure_probe_not_model_evaluation','execution':'inside-agent-container'}


class RobotWorldNativeBudgetProbeAgent(RobotWorldNativeProbeAgent):
    @staticmethod
    def name(): return 'robotworld-native-budget-probe'

    async def run(self,instruction,environment,context):
        logs=str(self.environment_logs_dir)
        await self.exec_as_agent(environment,command='python -m agents.native_probe --complete-episode --output '+shlex.quote(logs+'/robotworld'),
                                env={'PYTHONPATH':REMOTE},timeout_sec=3600)
        context.metadata={'classification':'infrastructure_probe_not_model_evaluation','execution':'inside-agent-container'}
