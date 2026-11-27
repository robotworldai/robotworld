"""Experimental Claude Code pilot using an explicitly selected Messages-compatible model.
No provider conversion is hidden here. No model fallback is configured.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import time
import signal

PROMPT = '''You control a robot through the RobotWorld MCP server. First call describe and observe.
Read the task instruction and every action's units, frame, absolute/delta semantics,
repeat semantics and limits. Execute physically plausible native actions to complete the task.
An action chunk is an ordered list of native commands. Read describe for each command's
repeat/duration semantics; some planner commands consume a variable number of native steps.
Use the robot tools actually listed in this run; never invent an unavailable execute tool. Native tools return fresh observations automatically. With the generic execute interface, call observe after execution. Inspect observations between chunks. A successful API call is not task success.
Only the environment's success flag establishes completion. Stop at terminal=true.
Do not claim success based on text or a plan. If your previous response ended while the
simulation remains active, continue controlling it. You have no simulator filesystem,
reset, hidden state, or direct physics API. Shell is available for offline computation and permitted observation processing. Never access hidden simulator state. Use coding_control only if describe lists it; its runtime schema is authoritative.
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--claude', default='claude');p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-continuations',type=int,default=20)
    p.add_argument('--wall-timeout',type=int,default=43200)
    p.add_argument('--instruction-file',type=Path)
    a=p.parse_args()
    instruction=a.instruction_file.read_text() if a.instruction_file else ''
    for key in ('ANTHROPIC_BASE_URL','ANTHROPIC_API_KEY','WORLD_EPISODE_URL','WORLD_EPISODE_TOKEN'):
        if not os.environ.get(key):p.error(f'{key} is required')
    from environment.clients.episode import EpisodeClient
    client=EpisodeClient();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    instruction=client.prepare_instruction(instruction,out)
    system_prompt=client.system_instruction(PROMPT,out)
    env=os.environ.copy();env['WORLD_AGENT_OBSERVATIONS']=str(out/'observations');env['CLAUDE_CONFIG_DIR']=str(out/'config')
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC']='1'
    root=Path(__file__).resolve().parents[2]
    env['PYTHONPATH']=str(root)
    config=out/'mcp.json'
    # Credentials remain inherited environment variables, never serialized here.
    config.write_text(json.dumps({'mcpServers':{'robotworld':{'command':sys.executable,'args':['-m','environment.bridges.mcp']}}},indent=2))
    session=str(uuid.uuid4())
    base=[a.claude,'--bare','-p','--model',a.model,'--tools','Bash,Read,Write,Edit,Glob,Grep',
          '--allowedTools','Bash,Read,Write,Edit,Glob,Grep,mcp__robotworld__*','--strict-mcp-config','--mcp-config',str(config),
          '--output-format','stream-json','--verbose','--append-system-prompt',system_prompt]
    (out/'configuration.json').write_text(json.dumps({'harness':'claude-code','interface_profile':os.environ.get('WORLD_AGENT_INTERFACE','native-feedback-v1'),'model':a.model,
        'version':subprocess.check_output([a.claude,'--version'],text=True).strip(),
        'mode':'persistent_episode','max_continuations':a.max_continuations,
        'builtin_tools':['Bash','Read','Write','Edit','Glob','Grep'],'code_control':'runtime_describe','task_instruction':instruction,
        'observation_window':{'enabled':True,'policy':'latest RobotWorld observation packet; other images preserved', 'request_counts':'events/model-image-window.jsonl'}},indent=2))
    from environment.runtime.messages_window_proxy import MessagesWindowProxy
    relay = MessagesWindowProxy(env['ANTHROPIC_BASE_URL'], env['ANTHROPIC_API_KEY'], out/'events/model-image-window.jsonl')
    with relay:
        env['ANTHROPIC_BASE_URL'] = relay.base_url
        run_episode(a, client, out, env, base, session, instruction)

def run_episode(a, client, out, env, base, session, instruction):
    reason='continuation_budget'
    deadline=time.monotonic()+a.wall_timeout
    for turn in range(a.max_continuations+1):
        if time.monotonic()>=deadline:reason='harness_timeout';break
        status=client.request('status')
        if status['terminal']:reason='environment_terminal';break
        command=base+(['--session-id',session] if turn==0 else ['--resume',session])
        command+=[('Complete the task. Start with describe and observe.\n'+instruction) if turn==0 else
                  'The environment is still active. Query observe and continue using robot actions.']
        with (out/f'turn-{turn:03d}.jsonl').open('w') as log, (out/f'turn-{turn:03d}.stderr').open('w') as err:
            proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=err,start_new_session=True)
            try:
                proc.wait(timeout=max(.1,deadline-time.monotonic()))
                r=proc
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                reason='harness_timeout';break
        if r.returncode:reason='harness_error';break
    (out/'summary.json').write_text(json.dumps({'harness_stop':reason,'environment':client.request('status')},indent=2))

if __name__=='__main__':main()
