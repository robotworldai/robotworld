"""Migration must preserve bytes and reject a misleading missing/foreign link."""
import json
import sys
from pathlib import Path
import pytest
from environment.datasets import centralize

@pytest.fixture
def layout(tmp_path,monkeypatch):
 monkeypatch.setattr(centralize,'WORLD',tmp_path)
 monkeypatch.setattr(centralize,'mappings',lambda:[{'source':'legacy/assets','target':'Assets/demo/data'}])
 report=tmp_path/'report.json'
 def run(action):
  monkeypatch.setattr(sys,'argv',['centralize',action,'--report',str(report)])
  centralize.main()
  return json.loads(report.read_text())
 return tmp_path,run

def test_move_and_repeat_preserve_nested_bytes_and_internal_symlink(layout):
 root,run=layout;source=root/'legacy/assets';source.mkdir(parents=True)
 (source/'robot').mkdir();(source/'robot/mesh.bin').write_bytes(bytes(range(256))*64)
 (source/'mesh-alias').symlink_to('robot/mesh.bin')
 before=centralize.digest_tree(source)
 result=run('apply');target=root/'Assets/demo/data'
 assert source.is_symlink() and source.resolve()==target
 assert centralize.digest_tree(target)==before==result['mappings'][0]['before']
 assert run('apply')['mappings'][0]['status']=='already_migrated'
 assert run('verify')['mappings'][0]['linked_to_canonical']

def test_dangling_canonical_link_is_not_success(layout):
 root,run=layout;source=root/'legacy/assets';source.parent.mkdir()
 source.symlink_to('../Assets/demo/data',target_is_directory=True)
 with pytest.raises(FileNotFoundError):run('apply')
 with pytest.raises(SystemExit):run('verify')

def test_foreign_link_is_not_repointed(layout):
 root,run=layout;source=root/'legacy/assets';source.parent.mkdir()
 foreign=root/'unrelated';foreign.mkdir();(foreign/'keep').write_text('unchanged')
 source.symlink_to(foreign)
 with pytest.raises(RuntimeError):run('apply')
 assert source.resolve()==foreign and (foreign/'keep').read_text()=='unchanged'

def test_failed_link_creation_restores_original_directory(layout,monkeypatch):
 root,run=layout;source=root/'legacy/assets';source.mkdir(parents=True);(source/'keep').write_text('data')
 def fail(*args,**kwargs):raise PermissionError('simulated link creation error')
 monkeypatch.setattr(Path,'symlink_to',fail)
 with pytest.raises(PermissionError):run('apply')
 assert source.is_dir() and not source.is_symlink() and (source/'keep').read_text()=='data'
 assert not (root/'Assets/demo/data').exists()
