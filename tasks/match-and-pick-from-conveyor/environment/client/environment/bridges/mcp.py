"""MCP stdio bridge. Install mcp>=1.20,<2 in the agent environment."""
import asyncio
import json
import copy
import base64
import hashlib
from pathlib import Path
import os
import threading
import uuid
import urllib.error
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent
from environment.clients.episode import EpisodeClient

client = None

def episode_client():
    global client
    if client is None:
        client = EpisodeClient()
    return client
server = Server('robotworld')
_action_lock = threading.Lock()
_native_specs = None


def interface_profile():
    value = os.environ.get('WORLD_AGENT_INTERFACE', 'native-feedback-v1')
    if value not in {'native-feedback-v1', 'generic-v1'}:
        raise ValueError('Unknown WORLD_AGENT_INTERFACE')
    return value


def native_specs():
    global _native_specs
    if _native_specs is None:
        specs = episode_client().request('describe')['tools']
        _native_specs = {}
        for spec in specs:
            name = spec['name']
            if name == 'observe':
                continue  # Read-only episode observation replaces legacy query tool.
            if name in {'describe','status','execute','receipt'} or name in _native_specs:
                raise ValueError('Native tool collides with episode protocol: ' + name)
            _native_specs[name] = spec
    return _native_specs


def native_call(name, arguments):
    # One serialized native command, translated to the unchanged episode protocol.
    # Never retry POST: uncertain execution must be resolved using its receipt.
    with _action_lock:
        c = episode_client()
        status = c.request('status')
        request_id = 'native-' + uuid.uuid4().hex
        request = dict(request_id=request_id, episode_id=status['episode_id'],
                       observation_id=status['observation_id'],
                       commands=[dict(tool=name, arguments=arguments)])
        try:
            result = c.request('execute', request)
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                detail = json.loads(exc.read().decode()).get('error', 'Action rejected before execution')
                raise ValueError(detail) from exc
            raise RuntimeError('Action response unavailable; query receipt with request_id=' + request_id) from exc
        except Exception as exc:
            raise RuntimeError('Action response unavailable; do not repeat the action. '
                'Query receipt with request_id=' + request_id + '. Cause: ' + type(exc).__name__) from exc
        try:
            packet = c.request('observe')
            if packet['observation_id'] != result['observation_id']:
                result['observation_error'] = 'Observation advanced after action; query observe before acting.'
            else:
                result['observation'] = packet['observation']
        except Exception as exc:
            # Preserve a known execution receipt even if rendering/query fails.
            result['observation_error'] = type(exc).__name__ + '; query observe, do not repeat action.'
        return result


@server.list_tools()
async def tools():
    generic = [Tool(name='describe',description='Read the task and native action schemas, units, frames and control frequency before acting.',inputSchema={'type':'object','properties':{},'additionalProperties':False}),
            Tool(name='observe',description='Read current RGB images and public proprioception. Does not advance time. History sampling and camera availability are specified by describe; query calls do not advance or append history.',inputSchema={'type':'object','properties':{},'additionalProperties':False}),
            Tool(name='status',description='Read actual control steps and environment terminal status. Textual claims do not change success.',inputSchema={'type':'object','properties':{},'additionalProperties':False}),
            Tool(name='execute',description='Execute an ordered native action chunk. Read describe for commands. Request IDs are idempotent. Stop automatically at native terminal or step budget. Query observe after execution.',inputSchema={
                'type':'object','properties':{'request_id':{'type':'string'},'episode_id':{'type':'string'},'observation_id':{'type':'string'},'commands':{'type':'array','minItems':1,'maxItems':256,'items':{'type':'object','properties':{'tool':{'type':'string'},'arguments':{'type':'object'}},'required':['tool','arguments'],'additionalProperties':False}}},'required':['request_id','episode_id','observation_id','commands'],'additionalProperties':False}),
            Tool(name='receipt',description='Query an execution receipt after a timeout. Never resend an uncertain action with a new ID.',inputSchema={'type':'object','properties':{'request_id':{'type':'string'}},'required':['request_id'],'additionalProperties':False})]

    if interface_profile() == 'generic-v1':
        return generic
    specifications = await asyncio.to_thread(native_specs)
    return [t for t in generic if t.name != 'execute'] + [
        Tool(name=name, description=spec.get('description','') +
             '\nReturns fresh public observation images after execution. All motion goes through the episode service.',
             inputSchema=spec['inputSchema']) for name, spec in specifications.items()]

def publish_observation(result):
    """Make only received public observations available to offline shell tools."""
    destination = os.environ.get('WORLD_AGENT_OBSERVATIONS')
    observation = result.get('observation')
    sensor_files = observation.pop('sensor_files', []) if isinstance(observation, dict) else []
    if not destination or not isinstance(observation, dict):
        if sensor_files:
            result['sensor_export_error'] = 'Agent observation export directory is not configured'
        return
    root = Path(destination)
    folder = root / str(int(result['control_step']))
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'state.json').write_text(json.dumps({k:v for k,v in observation.items() if k!='images'}, indent=2))
    files = []
    for img in observation.get('images', []):
        suffix = {'image/jpeg':'.jpg', 'image/png':'.png'}.get(img['mime_type'])
        if suffix is None:
            continue
        camera = hashlib.sha256(img['camera'].encode()).hexdigest()[:12]
        image_folder = root / str(int(img['control_step']))
        image_folder.mkdir(parents=True, exist_ok=True)
        path = image_folder / (camera + suffix)
        path.write_bytes(base64.b64decode(img['base64'], validate=True))
        files.append({'camera':img['camera'],'control_step':img['control_step'],'path':str(path)})
    (folder/'images.json').write_text(json.dumps(files,indent=2))
    sensors = []
    for sensor in sensor_files:
        if sensor['mime_type'] != 'application/x-npy':
            raise ValueError('Unsupported public sensor file type')
        # Never use a supplied filename or path, including camera names, as a path.
        sensor_folder = root / str(int(sensor['control_step']))
        sensor_folder.mkdir(parents=True, exist_ok=True)
        path = sensor_folder / (hashlib.sha256(sensor['sensor'].encode()).hexdigest()[:12]+'.npy')
        path.write_bytes(base64.b64decode(sensor['base64'], validate=True))
        sensors.append({k:v for k,v in sensor.items() if k!='base64'} | {'path':str(path)})
    result['local_observation_files'] = {'state':str(folder/'state.json'),'images':files,'sensors':sensors}


@server.call_tool()
async def call(name, arguments):
    if interface_profile() == 'native-feedback-v1' and name not in {'describe','observe','status','receipt'}:
        specifications = await asyncio.to_thread(native_specs)
        if name not in specifications:
            raise ValueError('Unknown tool')
        result = await asyncio.to_thread(native_call, name, arguments)
    else:
        allowed = {'describe','observe','status','receipt'}
        if interface_profile() == 'generic-v1': allowed.add('execute')
        if name not in allowed: raise ValueError('Unknown tool')
        path = 'receipts/'+arguments['request_id'] if name=='receipt' else name
        result = await asyncio.to_thread(episode_client().request, path, arguments if name=='execute' else None)
    result = copy.deepcopy(result)
    publish_observation(result)
    images = result.get('observation',{}).pop('images',[])
    blocks = [TextContent(type='text',text=json.dumps(result))]
    for img in images:
        blocks.extend([TextContent(type='text',text=f"Camera {img['camera']}, control step {img['control_step']}"),
                       ImageContent(type='image',data=img['base64'],mimeType=img['mime_type'])])
    return blocks

async def main():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())

if __name__=='__main__':asyncio.run(main())
