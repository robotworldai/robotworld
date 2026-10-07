"""Handoff exports must retain Codex source resources, not simulator assets."""
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'package_code', Path(__file__).resolve().parents[1]/'scripts/package_code.py')
package_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_code)


class PackageCodeTests(unittest.TestCase):
    def test_codex_assets_survive_export_without_leaking_runtime_files(self):
        keep = [
            'codex/codex-rs/core/assets/agent/agent_names.txt',
            'codex/codex-rs/core/assets/tools/apply_patch.lark',
            'codex/codex-rs/login/src/assets/success.html',
            'codex/codex-rs/skills/src/assets/samples/imagegen/SKILL.md',
            'codex/codex-rs/skills/src/assets/samples/imagegen/assets/imagegen.png',
            'codex/codex-rs/tui/assets/themes/ada.tmTheme',
        ]
        exclude = [
            'third_party/benchmarks/example/assets/scene.usd',
            'third_party/benchmarks/example/assets/metadata.json',
            'third_party/benchmarks/example/checkout/assets/object.usdz',
            'codex/.git/config',
            'codex/target/output',
            'codex/codex-rs/core/assets/auth.json',
            'codex/codex-rs/core/assets/.env',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'
            output = Path(tmp)/'export'
            for rel in keep + exclude + ['robotworld-scoring-guide.html']:
                path = root/rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture\n')
            with patch.object(package_code, 'ROOT', root), patch('sys.argv', [
                    'package_code.py', '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                package_code.main()
            manifest = json.loads((output/'code-export.json').read_text())
            exported = {r['path'] for r in manifest['files']}
            for rel in keep:
                self.assertIn(rel, exported)
                self.assertEqual((output/rel).read_bytes(), (root/rel).read_bytes())
            for rel in exclude:
                self.assertNotIn(rel, exported)
                self.assertFalse((output/rel).exists())


if __name__ == '__main__':
    unittest.main()
