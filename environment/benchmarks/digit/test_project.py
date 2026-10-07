import json
from . import project


def test_kit6_missing_asset_root_is_resolved_without_changing_task_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(project, 'ROOT', tmp_path)
    path = tmp_path / 'assets/Isaac/Environments/example.hdr'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'fixture')
    cfg = {'light': {'texture_file': 'None/Isaac/Environments/example.hdr'}, 'mass': 4.2}
    project.localize(cfg)
    assert cfg == {'light': {'texture_file': str(path)}, 'mass': 4.2}


def test_no_invented_binary_success():
    assert project.success(None) is None
    assert "NO held box" in project.instruction("T13", None)


def test_pinned_lab_not_base_image_overlay():
    config = json.loads((project.ROOT / "project.json").read_text())
    assert (project.ROOT / "checkout/VERSION").read_text().strip() == "2.3.2"
    assert all("digit/checkout/source" in p for p in config["runtime_python_paths"])
    row = config["tasks"]["T13"]
    assert row["steps"] * row["physics_dt"] * row["decimation"] == 14
