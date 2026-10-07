"""Regression checks for the A1 compatibility preprocessor's physical invariants."""
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from environment.benchmarks.robot_lab.asset_compat import prepare_urdf


SOURCE = (Path(__file__).resolve().parents[2] / 'third_party/benchmarks/robot_lab/checkout'
          / 'source/robot_lab/data/Robots/Unitree/A1/a1_description/urdf/a1.urdf')


def test_protected_edges_and_original_source_preserved(tmp_path):
    before = SOURCE.read_bytes()
    def inspect_merge_input(src, dst):
        root = ET.parse(src).getroot()
        assert not any(j.get('dont_collapse') == 'true' for j in root.findall('joint'))
        assert all(Path(m.get('filename')).is_file() for m in root.iter('mesh'))
        shutil.copyfile(src, dst)
    result = prepare_urdf(SOURCE, tmp_path, inspect_merge_input)
    root = ET.parse(result).getroot()
    feet = [j for j in root.findall('joint') if j.get('dont_collapse') == 'true']
    assert len(feet) == 4 and all(j.get('type') == 'fixed' for j in feet)
    assert len(root.findall("material[@name='grey']")) == 1
    assert root.find("material[@name='grey']/color").get('rgba') == '0.2 0.2 0.2 1.0'
    assert SOURCE.read_bytes() == before


@pytest.mark.parametrize('damage, message', [('mass', 'mass changed'), ('limit', 'limit changed')])
def test_rejects_physical_changes(tmp_path, damage, message):
    def corrupt_merge(src, dst):
        tree = ET.parse(src)
        if damage == 'mass':
            tree.find('link/inertial/mass').set('value', '900')
        else:
            tree.find("joint[@type='revolute']/limit").set('effort', '900')
        tree.write(dst)
    with pytest.raises(ValueError, match=message):
        prepare_urdf(SOURCE, tmp_path, corrupt_merge)


def test_runtime_profile_keeps_native_task():
    from environment.runtime.native_project_launch import load_project
    _, original = load_project('robot_lab')
    _, fixed = load_project('robot_lab', 'a1-feet')
    assert fixed['tasks'] == original['tasks']
    assert fixed['commit'] == original['commit']
    assert fixed['environment']['WORLD_A1_REIMPORT'] == '1'
    assert 'not official-equivalent' in fixed['runtime']
