"""Limited, schema-validated ATIF export of observable custom-agent events.

Runs on the Harbor host after native logs are synced. Raw archives stay intact.
No hidden reasoning, token usage, reward, or absent tool reply is inferred.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from harbor.models.trajectories.trajectory import Trajectory


class Exporter:
    def __init__(self, logs, harness, model):
        self.logs=Path(logs)
        self.native=self.logs/'robotworld'
        self.harness=harness;self.model=model;self.session=None
        self.steps=[];self.calls={};self.sources=[];self.skipped={}
        self.media=self.logs/'trajectory-media'

    def add(self, source, message='', timestamp=None, **fields):
        step={'step_id':len(self.steps)+1,'source':source,'message':message,**fields}
        if timestamp:step['timestamp']=timestamp
        self.steps.append(step)
        return step

    def skip(self, kind):self.skipped[kind]=self.skipped.get(kind,0)+1

    def rows(self,path):
        path=Path(path)
        self.sources.append(str(path.relative_to(self.logs)))
        with path.open() as reader:
            for number,line in enumerate(reader,1):
                if not line.strip():continue
                try:yield number,json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'Incomplete/invalid native JSONL at {path.name}:{number}') from error

    def image(self,url=None,source=None):
        if source and source.get('type')=='base64':
            mime=source['media_type'];data=source['data']
        elif isinstance(url,str) and url.startswith('data:image/') and ';base64,' in url:
            header,data=url.split(',',1);mime=header[5:].split(';')[0]
        else:
            self.skip('unsupported_image_reference')
            return {'type':'text','text':'[Image reference not converted; see the original event archive.]'}
        extension={'image/jpeg':'.jpg','image/png':'.png','image/webp':'.webp','image/gif':'.gif'}.get(mime)
        if extension is None:raise ValueError('Unsupported image MIME type in native trace')
        raw=base64.b64decode(data,validate=True)
        self.media.mkdir(exist_ok=True)
        filename=hashlib.sha256(raw).hexdigest()+extension
        target=self.media/filename
        if not target.exists():target.write_bytes(raw)
        return {'type':'image','source':{'media_type':mime,'path':'trajectory-media/'+filename}}

    def parts(self, content):
        if isinstance(content,str):return content
        parts=[]
        for block in content or []:
            if not isinstance(block,dict):
                parts.append({'type':'text','text':json.dumps(block,ensure_ascii=False)});continue
            kind=block.get('type')
            if kind in ('text','inputText','input_text','output_text'):
                parts.append({'type':'text','text':block.get('text','')})
            elif kind in ('inputImage','input_image','image_url'):
                url=block.get('imageUrl',block.get('image_url',block.get('url')))
                if isinstance(url,dict):url=url.get('url')
                parts.append(self.image(url=url))
            elif kind=='image':parts.append(self.image(url=block.get('url'),source=block.get('source')))
            elif kind in ('thinking','redacted_thinking','reasoning'):
                self.skip('reasoning_not_exported')
            elif kind not in ('tool_use','tool_result'):
                self.skip('unknown_content:'+str(kind))
        return parts or ''

    def tool(self,step,call_id,name,arguments):
        if call_id in self.calls:raise ValueError('Duplicate native tool call ID: '+str(call_id))
        if isinstance(arguments,str):arguments=json.loads(arguments)
        if not isinstance(arguments,dict):raise ValueError('Tool arguments must be an object')
        step.setdefault('tool_calls',[]).append({'tool_call_id':call_id,'function_name':name,'arguments':arguments})
        self.calls[call_id]=step

    def reply(self,call_id,content,extra=None):
        step=self.calls.get(call_id)
        if step is None:
            self.skip('unmatched_tool_result')
            self.add('user',self.parts(content),extra={'unmatched_native_tool_call_id':call_id})
            return
        observation=step.setdefault('observation',{'results':[]})
        result={'source_call_id':call_id,'content':self.parts(content)}
        if extra:result['extra']=extra
        observation['results'].append(result)

    def codex(self):
        pending={};seen_messages=set()
        source=self.native/'events/codex.jsonl'
        for number,record in self.rows(source):
            payload=record.get('payload',{});kind=record.get('kind');method=payload.get('method')
            params=payload.get('params',{});stamp=record.get('timestamp')
            origin={'native_file':str(source.relative_to(self.logs)),'native_line':number}
            if kind=='client_to_server' and method=='thread/start':
                self.model=params.get('model',self.model)
                instructions='\n'.join(params[k] for k in ('baseInstructions','developerInstructions') if params.get(k))
                if instructions:self.add('system',instructions,timestamp=stamp,extra=origin)
            elif kind=='client_to_server' and method=='turn/start':
                self.session=params.get('threadId',self.session)
                self.add('user',self.parts(params.get('input',[])),timestamp=stamp,extra=origin)
            elif kind=='server_to_client' and method=='item/tool/call':
                step=self.add('agent','',timestamp=stamp,extra=origin)
                self.tool(step,params['callId'],params['tool'],params['arguments'])
                pending[payload['id']]=params['callId']
            elif kind=='client_to_server' and 'id' in payload and payload['id'] in pending and 'result' in payload:
                result=payload['result'];call_id=pending.pop(payload['id'])
                self.reply(call_id,result.get('contentItems',[]),{'native_success':result.get('success'),'native_line':number})
            elif kind=='server_to_client' and method=='item/completed':
                item=params.get('item',{});item_kind=item.get('type')
                if item_kind=='agentMessage' and item.get('id') not in seen_messages:
                    seen_messages.add(item.get('id'));self.add('agent',item.get('text',''),timestamp=stamp,extra=origin)
                elif item_kind=='reasoning':self.skip('reasoning_not_exported')
                # User input and dynamic tool results already come from authoritative RPC.
                elif item_kind not in ('userMessage','dynamicToolCall'):self.skip('unconverted_codex_item:'+str(item_kind))
        for call_id in pending.values():self.calls[call_id].setdefault('extra',{})['tool_result_missing']=True

    def claude(self):
        instruction=self.logs/'task-instruction.txt'
        if instruction.exists():
            self.sources.append(instruction.name)
            self.add('user',instruction.read_text(),extra={'native_file':instruction.name,'scope':'task instruction; CLI continuation prompts are not fully represented'})
        messages={};seen_parts={}
        files=sorted(self.native.glob('turn-*.jsonl'))
        if not files:raise ValueError('No Claude stream JSONL files found')
        for source in files:
            for number,row in self.rows(source):
                self.session=row.get('session_id',self.session)
                kind=row.get('type');message=row.get('message',{})
                if kind=='system':
                    if row.get('subtype')=='init':self.model=row.get('model',self.model)
                    continue
                if kind=='assistant':
                    identity=(self.session,message.get('id'))
                    if identity[1] is None:identity=(str(source),number)
                    if identity not in messages:
                        messages[identity]=self.add('agent',[],timestamp=row.get('timestamp'),extra={
                            'native_file':str(source.relative_to(self.logs)),'native_line':number,'native_message_id':message.get('id')})
                        seen_parts[identity]=set()
                    step=messages[identity]
                    for block in message.get('content',[]):
                        digest=hashlib.sha256(json.dumps(block,sort_keys=True).encode()).hexdigest()
                        if digest in seen_parts[identity]:continue
                        seen_parts[identity].add(digest)
                        if block.get('type')=='tool_use':self.tool(step,block['id'],block['name'],block['input'])
                        else:
                            parts=self.parts([block])
                            if parts:step['message'].extend(parts if isinstance(parts,list) else [{'type':'text','text':parts}])
                    if message.get('model') and message['model'] != '<synthetic>':
                        self.model=message['model']
                    if row.get('is_api_error_message'):
                        step.setdefault('extra', {})['native_api_error_message'] = True
                elif kind=='user':
                    content=message.get('content',[])
                    if isinstance(content,str):self.add('user',content);continue
                    other=[]
                    for block in content:
                        if block.get('type')=='tool_result':
                            self.reply(block['tool_use_id'],block.get('content',[]),{'native_is_error':bool(block.get('is_error',False))})
                        else:other.append(block)
                    if other:self.add('user',self.parts(other),timestamp=row.get('timestamp'))
                elif kind not in ('result','stream_event'):self.skip('unconverted_claude_event:'+str(kind))
        for step in self.steps:
            if step['message']==[]:step['message']=''

    def export(self):
        if self.harness=='codex':self.codex()
        elif self.harness in ('claude-code','claude_code'):self.claude()
        else:raise ValueError('Only codex and claude-code native formats are supported')
        if not self.steps:raise ValueError('No observable conversation events; refusing a fabricated trajectory')
        trajectory=Trajectory.model_validate({'schema_version':'ATIF-v1.7','session_id':self.session,
            'agent':{'name':'robotworld-'+self.harness,'version':'episode-v1-harbor-1','model_name':self.model},
            'steps':self.steps,'notes':'Limited observable-event export. Original native traces remain authoritative. No hidden reasoning or missing events are reconstructed.',
            'extra':{'exporter':'robotworld-observable-atif-v1','lossless':False,
                'source_files':self.sources,'skipped_events':self.skipped,
                'simulation_artifacts':'../artifacts/simulation',
                'limitations':['No hidden reasoning export','No inferred token/cost/reward metrics',
                    'ATIF steps represent observable events, not simulation control steps or guaranteed model-call boundaries',
                    'Claude CLI continuation prompts and internal compaction are not fully represented',
                    'Only RobotWorld dynamic tool RPC is converted for Codex; raw trace preserves other runtime events']}})
        target=self.logs/'trajectory.json';temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(trajectory.to_json_dict(),ensure_ascii=False,indent=2))
        temporary.replace(target)
        return target


def export_trajectory(logs_dir, harness, model_name=None):
    """Host-side post-sync hook: return validated logs_dir/trajectory.json."""
    return Exporter(logs_dir,harness,model_name).export()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--logs-dir',type=Path,required=True)
    parser.add_argument('--harness',choices=['codex','claude-code'],required=True);parser.add_argument('--model')
    args=parser.parse_args();print(export_trajectory(args.logs_dir,args.harness,args.model))
