"""Source-built Codex neutral-episode pilot with an explicit third-party provider."""
from contextlib import ExitStack
from environment.runtime.observation_window_proxy import ObservationWindowProxy
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from environment.runtime.codex_session import CodexSession
from environment.clients.episode import EpisodeClient
from environment.bridges.mcp import tools, call, interface_profile
from harnesses.claude_code.run import PROMPT


def timeout_stop_reason(deadline, now=None):
 """Only exhaustion of the trial clock is an agent budget stop."""
 now = time.monotonic() if now is None else now
 return 'harness_timeout' if deadline is not None and now >= deadline else 'harness_error'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--wall-timeout',type=int,default=43200)
 p.add_argument('--instruction-file',type=Path)
 a=p.parse_args();root=Path(__file__).resolve().parents[2]
 instruction=a.instruction_file.read_text() if a.instruction_file else ''
 a.output.mkdir(parents=True,exist_ok=False)
 os.environ['WORLD_AGENT_OBSERVATIONS']=str(a.output.resolve()/'observations')
 config=json.loads(Path(os.environ['WORLD_CODEX_PROVIDER_CONFIG']).read_text())
 overrides=list(config['overrides']);manifest=Path(config['manifest'])
 client=EpisodeClient();reason='harness_error';error=None;deadline=None
 (a.output/'configuration.json').write_text(json.dumps({'harness':'source-built-codex','interface_profile':interface_profile(),'model':a.model,'provider':config['provider'],'code_control':'runtime_describe','task_instruction':instruction,'observation_window':{'enabled':bool(config.get('observation_window')),'policy':'latest RobotWorld packet; full original images archived','request_counts':'events/model-image-window.jsonl'}},indent=2))
 try:
  instruction=client.prepare_instruction(instruction,a.output)
  system_prompt=client.system_instruction(PROMPT,a.output)
  with ExitStack() as stack:
   window=config.get('observation_window')
   if not window:raise ValueError('Provider config must specify observation_window upstream_base_url and api_key_env')
   relay=stack.enter_context(ObservationWindowProxy(window['upstream_base_url'],window['api_key_env'],a.output/'events/model-image-window.jsonl'))
   overrides.extend([f"model_providers.{config['provider']}.base_url="+json.dumps(relay.base_url),f"model_providers.{config['provider']}.supports_websockets=false"])
   s=stack.enter_context(CodexSession(manifest,a.output,config_overrides=overrides,timeout=180))
   specs=[{'type':'function','name':x.name,'description':x.description,'inputSchema':x.inputSchema} for x in asyncio.run(tools())]
   r=s.rpc('thread/start',{'model':a.model,'modelProvider':config['provider'],'cwd':str(a.output.resolve()),'approvalPolicy':'never','sandbox':'workspace-write','ephemeral':True,'baseInstructions':system_prompt,'dynamicTools':specs,'config':{'features.shell_tool':True,'features.code_mode':False,'features.apps':False,'features.plugins':False,'web_search':'disabled'}})
   thread=r['thread']['id'];continuations=0;deadline=time.monotonic()+a.wall_timeout
   def start(text):s.rpc('turn/start',{'threadId':thread,'input':[{'type':'text','text':text,'text_elements':[]}]})
   start('Complete the task. Start with describe and observe.\n'+instruction)
   while time.monotonic()<deadline:
    m=s.receive(timeout=max(.1,deadline-time.monotonic()));method=m.get('method');params=m.get('params',{})
    if method=='item/tool/call':
     ok=True
     try:
      blocks=asyncio.run(call(params['tool'],params['arguments']))
      content=[{'type':'inputText','text':x.text} if x.type=='text' else {'type':'inputImage','imageUrl':'data:'+x.mimeType+';base64,'+x.data} for x in blocks]
     except Exception as e:ok=False;content=[{'type':'inputText','text':str(e)}]
     s.send({'id':m['id'],'result':{'success':ok,'contentItems':content}})
    elif method=='turn/completed':
     if params['turn']['status']!='completed':raise RuntimeError(str(params['turn'].get('error')))
     if client.request('status')['terminal']:reason='environment_terminal';break
     if continuations>=20:reason='continuation_budget';break
     continuations+=1;start('The environment is still active. Query observe and continue using robot actions.')
    elif 'id' in m and method:s.send({'id':m['id'],'error':{'code':-32601,'message':'Unsupported request'}})
   else:reason='harness_timeout'
 except TimeoutError as e:reason=timeout_stop_reason(deadline);error=str(e)
 except Exception as e:error=str(e)
 finally:
  (a.output/'summary.json').write_text(json.dumps({'harness_stop':reason,'error':error,'environment':client.request('status')},indent=2))

if __name__=='__main__':main()
