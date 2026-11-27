import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('make_job',Path(__file__).parents[1]/'scripts/make_harbor_job.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def root(tmp_path):
    task=tmp_path/'tasks/example';task.mkdir(parents=True)
    (task/'task.toml').write_text('version="1.0"\n')
    return tmp_path

def test_agent_model_independently_selected_without_task_changes(root):
    a=m.build_job(root,tasks=['example'],agent='codex',agent_import_path=None,model='openai/model-a')
    b=m.build_job(root,tasks=['example'],agent='claude-code',agent_import_path=None,model='anthropic/model-b')
    assert a['tasks']==b['tasks']
    assert a['agents'][0]['name']=='codex'
    assert b['agents'][0]['model_name']=='anthropic/model-b'
    assert (root/'tasks/example/task.toml').read_text()=='version="1.0"\n'

def test_custom_agent(root):
    job=m.build_job(root,tasks=['all'],agent=None,agent_import_path='my_agent:Agent',model='provider/model',attempts=3)
    assert job['agents']==[{'import_path':'my_agent:Agent','model_name':'provider/model'}]
    assert job['n_attempts']==3

def test_outside_task_rejected(root):
    with pytest.raises(ValueError):m.build_job(root,tasks=['../../outside'],agent='codex',agent_import_path=None,model='p/m')
