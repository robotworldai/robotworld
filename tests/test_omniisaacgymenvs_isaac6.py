"""CPU contract checks for the explicitly experimental T14 runtime."""
import json
from pathlib import Path
from environment.runtime.native_project_launch import load_project
from environment.benchmarks.omniisaacgymenvs import project, project_isaac6


def test_profile_does_not_replace_original_task_or_source():
    _,old=load_project('omniisaacgymenvs')
    _,new=load_project('omniisaacgymenvs','isaac6')
    for key in ['key','repository','commit','tasks','engine']:
        assert new[key]==old[key]
    assert new['adapter'].endswith('.project_isaac6')
    assert old['adapter'].endswith('.project')
    assert new['image']!=old['image']


def test_experimental_step_and_observation_are_original():
    assert project_isaac6.Simulator.step is project.Simulator.step
    assert project_isaac6.Simulator.observation is project.Simulator.observation
    assert project_isaac6.Simulator.result is project.Simulator.result
    assert project_isaac6.Simulator.reset is project.Simulator.reset


def test_legacy_runtime_code_matches_extraction_manifest():
    import hashlib
    root=project.SOURCE.parent/'compat'
    for row in json.loads((root/'vendor-provenance.json').read_text())['files']:
        assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']

