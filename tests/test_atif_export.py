import base64
import json
from pathlib import Path
import pytest
from agents.trajectory import export_trajectory
from harbor.models.trajectories.trajectory import Trajectory


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def test_codex_rpc_preserves_call_reply_image_and_omits_reasoning(tmp_path):
    image='data:image/png;base64,'+base64.b64encode(b'exact image bytes').decode()
    def event(direction,payload):return {'kind':direction,'payload':payload}
    path=tmp_path/'robotworld/events/codex.jsonl'
    write(path,[event('client_to_server',{'method':'turn/start','params':{'threadId':'s','input':[{'type':'text','text':'go'}]}}),
        event('server_to_client',{'id':7,'method':'item/tool/call','params':{'callId':'c','tool':'observe','arguments':{}}}),
        event('client_to_server',{'id':7,'result':{'success':True,'contentItems':[{'type':'inputText','text':'observed'},{'type':'inputImage','imageUrl':image}]}}),
        event('server_to_client',{'method':'item/completed','params':{'item':{'type':'reasoning','content':['not exported']}}}),
        event('server_to_client',{'method':'item/completed','params':{'item':{'id':'m','type':'agentMessage','text':'done'}}})])
    before=path.read_bytes();out=export_trajectory(tmp_path,'codex','model')
    data=json.loads(out.read_text());Trajectory.model_validate(data)
    assert 'not exported' not in out.read_text()
    assert data['steps'][1]['tool_calls'][0]['tool_call_id']=='c'
    imagepath=data['steps'][1]['observation']['results'][0]['content'][1]['source']['path']
    assert (tmp_path/imagepath).read_bytes()==b'exact image bytes'
    assert path.read_bytes()==before
    assert data['extra']['lossless'] is False


def test_claude_fragments_merge_and_result_matches_call(tmp_path):
    a={'type':'assistant','session_id':'s','message':{'id':'m','content':[{'type':'text','text':'look'}]}}
    b={'type':'assistant','session_id':'s','message':{'id':'m','content':[{'type':'tool_use','id':'c','name':'observe','input':{}}]}}
    result={'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'c','content':'ok'}]}}
    write(tmp_path/'robotworld/turn-000.jsonl',[a,b,b,result])
    data=json.loads(export_trajectory(tmp_path,'claude-code').read_text())
    assert len(data['steps'])==1
    assert len(data['steps'][0]['tool_calls'])==1
    assert data['steps'][0]['observation']['results'][0]['content']=='ok'
    Trajectory.model_validate(data)


def test_pending_call_is_not_invented_as_success(tmp_path):
    write(tmp_path/'robotworld/events/codex.jsonl',[{'kind':'server_to_client','payload':{'id':7,'method':'item/tool/call','params':{'callId':'c','tool':'execute','arguments':{}}}}])
    data=json.loads(export_trajectory(tmp_path,'codex').read_text())
    assert 'observation' not in data['steps'][0]
    assert data['steps'][0]['extra']['tool_result_missing'] is True


def test_invalid_trace_does_not_create_fake_trajectory(tmp_path):
    path=tmp_path/'robotworld/events/codex.jsonl';path.parent.mkdir(parents=True);path.write_text('{partial')
    with pytest.raises(ValueError):export_trajectory(tmp_path,'codex')
    assert not (tmp_path/'trajectory.json').exists()


def test_synthetic_error_does_not_overwrite_provider_model(tmp_path):
    import json
    from agents.trajectory import export_trajectory
    native=tmp_path/'robotworld';native.mkdir()
    rows=[{'type':'system','subtype':'init','model':'provider-model','session_id':'session'},
          {'type':'assistant','is_api_error_message':True,'message':{'id':'error','model':'<synthetic>',
            'content':[{'type':'text','text':'API error'}]}}]
    (native/'turn-000.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    data=json.loads(export_trajectory(tmp_path,'claude-code').read_text())
    assert data['agent']['model_name']=='provider-model'
    assert data['steps'][0]['extra']['native_api_error_message'] is True
