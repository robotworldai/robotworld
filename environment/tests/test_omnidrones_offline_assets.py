import hashlib
import json
from types import SimpleNamespace

import pytest

from environment.benchmarks.omnidrones import offline_assets


def test_only_default_path_changes_and_physics_arguments_survive(tmp_path, monkeypatch):
    data=b'original asset bytes'
    (tmp_path/'default_environment.usd').write_bytes(data)
    monkeypatch.setattr(offline_assets, 'FILES', {'default_environment.usd':hashlib.sha256(data).hexdigest()})
    calls=[]
    kit=SimpleNamespace(create_ground_plane=lambda *a,**kw:calls.append((a,kw)))
    offline_assets.install(kit,tmp_path,tmp_path/'report.json')
    options=dict(z_position=-.5,static_friction=.8,dynamic_friction=.7,restitution=.1,
                 improve_patch_friction=False,color=(1,0,0))
    kit.create_ground_plane('/World/ground',**options)
    assert calls[-1]==(('/World/ground',),dict(options,usd_path=str(tmp_path/'default_environment.usd')))
    kit.create_ground_plane('/World/other',usd_path='explicit.usd',**options)
    assert calls[-1][1]['usd_path']=='explicit.usd'
    assert json.loads((tmp_path/'report.json').read_text())['upstream_usd_unchanged']


def test_missing_or_changed_assets_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(offline_assets, 'FILES', {'default_environment.usd':'wrong'})
    kit=SimpleNamespace(create_ground_plane=lambda:None)
    original=kit.create_ground_plane
    with pytest.raises(RuntimeError):offline_assets.install(kit,tmp_path,tmp_path/'report.json')
    (tmp_path/'default_environment.usd').write_bytes(b'different')
    with pytest.raises(RuntimeError):offline_assets.install(kit,tmp_path,tmp_path/'report.json')
    assert kit.create_ground_plane is original
