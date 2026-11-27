"""Public static briefs reuse the same contracts as the episode adapters."""
def control_brief(benchmark):
    from environment.benchmarks.action_contracts import CONTRACTS, action_contract
    from environment.benchmarks.operating_brief import SCENES, operating_brief
    parts = ['## Robot and control reference\n']
    if benchmark in SCENES:
        parts.append(operating_brief(benchmark))
    if benchmark in CONTRACTS:
        parts.append(action_contract(benchmark))
    parts.append(
        'The public runtime describe response supplies the selected robot/controller profile, '
        'native action schemas and task-specific instruction. The RobotWorld Codex and Claude Code '
        'agents attach that response before the first model request. Other agents must read describe '
        'before acting. Do not guess missing joint order, dimensions, gains or coordinate frames. '
        'Use observe for measured state and images. Static examples describe requested targets, '
        'not guaranteed motion or a reference solution. Only tools in the runtime schema are enabled.'
    )
    return '\n\n'.join(parts) + '\n'
