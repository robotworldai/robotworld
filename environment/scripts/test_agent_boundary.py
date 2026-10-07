"""Real source-Codex smoke test: shell + file memory + synthetic dynamic tool.

No simulator is started and no task success is measured.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from environment.runtime.codex_session import CodexSession
from environment.runtime.isolated_codex import IsolatedCodex


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--codex-home',type=Path,required=True)
    a=p.parse_args()
    world=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=world/'codex',text=True).strip()
    manifest=world/'var/build/codex'/commit/'build.json'
    a.output.mkdir(parents=True,exist_ok=False)
    with IsolatedCodex(manifest,a.output.resolve(),a.codex_home.resolve(),world/'var/configs/models-direct.json') as relay:
        os.environ['WORLD_CODEX_SOCKET']=str(relay.socket_path)
        with CodexSession(manifest,a.output,timeout=180) as session:
            params={'cwd':'/workspace','approvalPolicy':'never','sandbox':'danger-full-access','ephemeral':True,
                'config':{'features.shell_tool':True,'features.code_mode':True,'features.apps':False,
                          'features.plugins':False,'features.multi_agent':False,'web_search':'disabled'},
                'dynamicTools':[{'name':'boundary_probe','description':'Synthetic integration probe; no robot connected.',
                    'inputSchema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}}]}
            (a.output/'thread-config.json').write_text(json.dumps(params,indent=2))
            thread=session.rpc('thread/start',params)['thread']['id']
            session.rpc('turn/start',{'threadId':thread,'input':[{'type':'text','text':
                'Short integration test, not a robot task. Use your shell/exec tool to run Python: assert /data, /runs, '
                '/behavior-src, /agent-bridge and /var/run/docker.sock do not exist; write /workspace/memory.json '
                'containing {"test":"ok"}; print the file. Then call boundary_probe with ok=true. Then finish. '
                'Do not inspect credentials or use network resources.', 'text_elements':[]}]})
            deadline=time.monotonic()+180
            called=False
            while time.monotonic()<deadline:
                msg=session.receive(deadline-time.monotonic())
                if msg.get('method')=='item/tool/call':
                    call=msg['params']
                    called=call['tool']=='boundary_probe' and call['arguments']=={'ok':True}
                    session.send({'id':msg['id'],'result':{'success':called,'contentItems':[
                        {'type':'inputText','text':'Synthetic probe complete; finish now.'}]}})
                elif msg.get('method')=='turn/completed':
                    break
                elif 'id' in msg and 'method' in msg:
                    session.send({'id':msg['id'],'error':{'code':-32601,'message':'Unsupported client RPC'}})
            memory=a.output/'agent-workspace/memory.json'
            report={'dynamic_tool_called':called,'shell_memory_written':memory.exists(),
                    'memory':json.loads(memory.read_text()) if memory.exists() else None}
            (a.output/'result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            if not called or report['memory']!={'test':'ok'}:
                raise RuntimeError('Source Codex capability smoke test failed; inspect events')


if __name__=='__main__':
    main()
