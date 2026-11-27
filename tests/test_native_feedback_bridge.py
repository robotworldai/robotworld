import asyncio
import json
import pytest
from environment.bridges import mcp


class Client:
    def __init__(self):
        self.calls=[]; self.steps=0; self.fail_post=False; self.fail_observe=False
        self.observation={'state':{'x':1},'images':[{'camera':'front','control_step':1,'mime_type':'image/png','base64':'YWJj'}]}
    def request(self,path,body=None):
        self.calls.append((path,body))
        if path=='describe': return {'tools':[{'name':'move_eef','description':'native delta','inputSchema':{'type':'object','properties':{'steps':{'type':'integer'}}}}]}
        if path=='execute':
            self.steps+=1
            if self.fail_post:raise TimeoutError()
            return {'observation_id':'e:1','request_id':body['request_id'],'control_step':1}
        if path=='observe':
            if self.fail_observe:raise TimeoutError()
            return {'observation_id':'e:1','observation':self.observation}
        return {'episode_id':'e','observation_id':'e:0'}


@pytest.fixture
def client(monkeypatch):
    c=Client(); monkeypatch.setattr(mcp,'client',c);monkeypatch.setattr(mcp,'_native_specs',None)
    monkeypatch.setenv('WORLD_AGENT_INTERFACE','native-feedback-v1')
    return c


def test_native_schema_and_feedback(client):
    tools=asyncio.run(mcp.tools())
    assert {t.name for t in tools}=={'describe','observe','status','receipt','move_eef'}
    assert next(t for t in tools if t.name=='move_eef').inputSchema==client.request('describe')['tools'][0]['inputSchema']
    blocks=asyncio.run(mcp.call('move_eef',{'steps':1}))
    assert client.steps==1 and any(b.type=='image' for b in blocks)
    request=next(body for path,body in client.calls if path=='execute')
    assert request['commands']==[{'tool':'move_eef','arguments':{'steps':1}}]
    assert client.observation['images']  # Formatting must not mutate cached observations.


def test_uncertain_post_never_retried(client):
    client.fail_post=True
    with pytest.raises(RuntimeError,match='Query receipt with request_id='):
        asyncio.run(mcp.call('move_eef',{'steps':1}))
    assert client.steps==1
    assert sum(path=='execute' for path,_ in client.calls)==1


def test_observation_failure_preserves_receipt(client):
    client.fail_observe=True
    blocks=asyncio.run(mcp.call('move_eef',{'steps':1}))
    result=json.loads(blocks[0].text)
    assert result['control_step']==1 and result['request_id'].startswith('native-')
    assert 'do not repeat' in result['observation_error']


def test_generic_profile_still_available(client,monkeypatch):
    monkeypatch.setenv('WORLD_AGENT_INTERFACE','generic-v1')
    assert {t.name for t in asyncio.run(mcp.tools())}=={'describe','observe','status','execute','receipt'}
    assert not client.calls


def test_unknown_tool_does_not_move(client):
    with pytest.raises(ValueError):asyncio.run(mcp.call('unknown',{}))
    assert client.steps==0


def test_shell_exports_only_received_public_observation(client,monkeypatch,tmp_path):
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS',str(tmp_path/'observations'))
    result={'control_step':1,'observation':client.observation}
    mcp.publish_observation(result)
    files=result['local_observation_files']
    from pathlib import Path
    assert json.loads(Path(files['state']).read_text())=={'state':{'x':1}}
    assert Path(files['images'][0]['path']).read_bytes()==b'abc'
    assert Path(files['images'][0]['path']).is_relative_to(tmp_path)


def test_public_sensor_export_does_not_leak_bytes_or_use_supplied_paths(client,monkeypatch,tmp_path):
    from pathlib import Path
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS',str(tmp_path))
    client.observation['sensor_files']=[{'sensor':'../../outside','control_step':1,
        'mime_type':'application/x-npy','units':'metres','base64':'YWJj'}]
    blocks=asyncio.run(mcp.call('move_eef',{'steps':1}))
    response=json.loads(blocks[0].text)
    path=Path(response['local_observation_files']['sensors'][0]['path'])
    assert path.is_relative_to(tmp_path) and path.read_bytes()==b'abc'
    assert 'sensor_files' not in response['observation']
    assert 'base64' not in response['local_observation_files']['sensors'][0]
    assert client.observation['sensor_files']  # Shared transport cache is intact.


def test_sensor_bytes_not_forwarded_without_export_directory(client,monkeypatch):
    monkeypatch.delenv('WORLD_AGENT_OBSERVATIONS',raising=False)
    client.observation['sensor_files']=[{'base64':'YWJj'}]
    response=json.loads(asyncio.run(mcp.call('move_eef',{'steps':1}))[0].text)
    assert 'sensor_files' not in response['observation']
    assert response['sensor_export_error']


def test_preexecution_http_rejection_is_not_uncertain_retry(client,monkeypatch):
    import io
    import urllib.error
    original=client.request
    def request(path,body=None):
        if path=='execute':
            client.calls.append((path,body))
            raise urllib.error.HTTPError('http://local/execute',400,'Bad Request',{},io.BytesIO(b'{"error":"Invalid normalized vector"}'))
        return original(path,body)
    client.request=request
    with pytest.raises(ValueError,match='Invalid normalized vector'):
        asyncio.run(mcp.call('move_eef',{'steps':1}))
    assert client.steps==0 and sum(p=='execute' for p,_ in client.calls)==1
