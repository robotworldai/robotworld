import argparse
import json
from pathlib import Path
import tomllib
import pytest
from environment.evaluation.runner import selected, run_command
from environment.runtime.model_config import filtered_config, prepare_model_home, toml_text

def test_provider_preserves_model_but_not_host_tools(tmp_path):
    config={'model':'model-x','model_provider':'gateway','mcp_servers':{'sim':{}},'sandbox_mode':'danger-full-access','model_providers':{'gateway':{'name':'Test','base_url':'http://localhost:8000/v1','wire_api':'responses','env_key':'TEST_API_KEY','requires_openai_auth':False}}}
    result=filtered_config(config,environ={'TEST_API_KEY':'synthetic-token'})
    assert set(result)=={'model','model_provider','model_providers'}
    assert result['model_providers']['gateway']['experimental_bearer_token']=='synthetic-token'
    assert 'env_key' not in result['model_providers']['gateway']
    assert tomllib.loads(toml_text(result))==result
    with pytest.raises(ValueError,match='Missing model credential'):
        filtered_config(config,environ={})
    source=tmp_path/'source';source.mkdir();(source/'config.toml').write_text(toml_text(result))
    home=tmp_path/'runtime';public=prepare_model_home(source,home,model='other-model')
    assert public['model']=='other-model'
    assert 'synthetic-token' not in json.dumps(public)
    assert (home/'config.toml').stat().st_mode & 0o777 == 0o600

def test_reject_silent_provider_fallback():
    with pytest.raises(ValueError):filtered_config({'model_provider':'unknown'})
    with pytest.raises(ValueError,match='Responses'):
        filtered_config({'model_provider':'x','model_providers':{'x':{'wire_api':'chat'}}})

def test_suite_task_mapping_and_native_horizons():
    rows=selected('robolab','cube-left,cube-front')
    assert len(rows)==2 and all(r['steps']==450 for r in rows)
    assert len(selected('robocasa',None))==10
    assert [len(selected(b,None)) for b in ('robodojo','behavior_1k','robocasa','robolab')]==[15,10,10,10]
    assert 'ToolOrganizationTask' in {r['task'] for r in selected('robolab',None)}
    assert 'ToolOrganizationBothTask' not in {r['task'] for r in selected('robolab',None)}
    assert selected('robocasa','28')[0]['alias_of']=='10'
    with pytest.raises(ValueError):selected('robolab','missing')
    a=argparse.Namespace(steps=None,timeout=7200,runtime='isaac601',seed=7,split='pretrain',episode_index=0)
    cmd=run_command('robolab',rows[0],a,Path('/out'),Path('/auth'))
    assert '--isaac601' in cmd and '--max-steps' not in cmd
    a.steps=600
    with pytest.raises(ValueError):run_command('robolab',rows[0],a,Path('/out'),Path('/auth'))
    cmd=run_command('robocasa',selected('robocasa','8')[0],a,Path('/out'),Path('/auth'))
    assert cmd[cmd.index('--horizon')+1]=='600'
