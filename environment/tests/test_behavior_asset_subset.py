import io,json,zipfile
from pathlib import Path
import pytest
from environment.datasets.behavior_1k import subset


def test_subset_does_not_extract_unselected_assets(monkeypatch,tmp_path):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        z.writestr('task/object.usd',b'original')
        z.writestr('unrelated/other.usd',b'excluded')
        z.writestr('../escape',b'bad')
    class LocalRange(io.BytesIO):
        transferred=0
        def __init__(self,url):super().__init__(data.getvalue())
    monkeypatch.setattr(subset,'RangeFile',LocalRange)
    idx=tmp_path/'index.json';idx.write_text(json.dumps({'url':'mock','revision':'pin'}))
    out=tmp_path/'assets'
    report=subset.fetch(idx,out,['task/object.usd'])
    assert (out/'task/object.usd').read_bytes()==b'original'
    assert not (out/'unrelated').exists()
    assert report['files'][0]['bytes']==8
    with pytest.raises(ValueError):subset.fetch(idx,out,['../escape'])
    assert not (tmp_path/'escape').exists()


def test_particle_conditions_include_disabled_and_saturation_dependencies(tmp_path):
    from environment.datasets.behavior_1k.systems import select_system_files
    import json
    taxonomy = tmp_path / 'taxonomy.json'
    taxonomy.write_text(json.dumps([
        {'name': 'sofa.n.01', 'categories': ['sofa'], 'abilities': {
            'particleRemover': {'conditions': {'dust.n.01': None, 'mud.n.01': [['saturated', 'water.n.01']]}}}},
        {'name': 'dust.n.01', 'abilities': {}, 'substances': ['dust']},
        {'name': 'mud.n.01', 'abilities': {}, 'substances': ['mud']},
        {'name': 'water.n.01', 'abilities': {}, 'substances': ['water']},
    ]))
    files = [{'path': p} for p in ['systems/dust/metadata.json', 'systems/mud/metadata.json',
                                  'systems/water/metadata.json', 'systems/dust/z/usd/z.usd',
                                  'systems/dust/a/usd/a.usd', 'systems/unrelated/metadata.json']]
    systems, selected = select_system_files({'sofa'}, taxonomy, files)
    assert systems == ['dust', 'mud', 'water']
    assert 'systems/dust/a/usd/a.usd' in selected
    assert 'systems/dust/z/usd/z.usd' not in selected
    assert 'systems/unrelated/metadata.json' not in selected
