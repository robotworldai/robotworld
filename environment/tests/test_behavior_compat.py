import pytest
from environment.benchmarks.behavior_1k.compat import apply_diff, transformed


def test_exact_patch_and_mismatched_context():
    patch = '--- a\n+++ b\n@@ -1,3 +1,3 @@\n first\n-old\n+new\n last\n'
    assert apply_diff('first\nold\nlast\n', patch) == 'first\nnew\nlast\n'
    with pytest.raises(ValueError, match='context mismatch'):
        apply_diff('first\nchanged\nlast\n', patch)


def test_reject_unreviewed_source_before_patch(tmp_path):
    path = tmp_path / 'upstream.py'
    path.write_text('changed = True\n')
    with pytest.raises(RuntimeError, match='Unreviewed upstream'):
        transformed(path, {'sha256_before': '0' * 64})


def test_reject_historical_renderer_override(monkeypatch, tmp_path):
    from environment.benchmarks.behavior_1k.compat import activate
    monkeypatch.setenv('WORLD_BEHAVIOR_RENDERER', 'RayTracedLighting')
    with pytest.raises(RuntimeError, match='Preserve the upstream renderer'):
        activate(tmp_path)
