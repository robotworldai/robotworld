"""Opt-in, hash-checked Isaac 6.0.1 adaptations; upstream files remain untouched."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent


def apply_diff(source, patch):
    """Apply exact unified hunks, refusing offsets, fuzz, or mismatched context."""
    lines = source.splitlines(keepends=True)
    output, cursor = [], 0
    chunks = patch.splitlines(keepends=True)
    i = 2
    while i < len(chunks):
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', chunks[i])
        if not match:
            raise ValueError('Invalid compatibility hunk')
        start = int(match[1]) - 1 if int(match[1]) else 0
        if start < cursor:
            raise ValueError('Overlapping compatibility hunks')
        output.extend(lines[cursor:start]); cursor = start; i += 1
        while i < len(chunks) and not chunks[i].startswith('@@ '):
            line = chunks[i]; i += 1
            if line[0] in ' -':
                if cursor >= len(lines) or lines[cursor] != line[1:]:
                    raise ValueError('Compatibility source context mismatch')
                cursor += 1
            if line[0] in ' +':
                output.append(line[1:])
            if line[0] not in ' +-':
                raise ValueError('Unsupported compatibility patch line')
    output.extend(lines[cursor:])
    return ''.join(output)


def transformed(path, record):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != record['sha256_before']:
        raise RuntimeError(f'Unreviewed upstream source: {path}')
    result = apply_diff(source.decode(), (ROOT / record['patch']).read_text())
    if hashlib.sha256(result.encode()).hexdigest() != record['sha256_after']:
        raise RuntimeError(f'Compatibility patch digest mismatch: {path}')
    return result


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        record = self.records[fullname]
        result = transformed(self.path, record)
        if fullname == 'omnigibson.simulator':
            # Keep __file__ upstream for metadata/materials; only Kit source lives outside.
            old = 'kit_file = Path(__file__).parent / kit_file_name'
            assert result.count(old) == 1
            result = result.replace(old, f'kit_file = Path({str(ROOT)!r}) / kit_file_name')
            if os.environ.get('WORLD_BEHAVIOR_RENDER_TRACE'):
                old = 'self._sim_context.render()'
                assert result.count(old) == 1
                result = result.replace(old,
                    '__import__("environment.benchmarks.behavior_1k.compat.render_diagnostics", '
                    'fromlist=["render"]).render(self._sim_context)')
        return compile(result, self.path, 'exec')


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, records):
        self.records = records

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.records:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError(fullname)
        loader = _Loader(fullname, spec.origin)
        loader.records = self.records
        spec.loader = loader
        return spec


def activate(output):
    if os.environ.get('WORLD_BEHAVIOR_RENDERER', 'upstream') != 'upstream':
        raise RuntimeError('Preserve the upstream renderer; the historical lighting override is no longer supported')
    if any(name == 'omnigibson' or name.startswith('omnigibson.') for name in sys.modules):
        raise RuntimeError('Activate compatibility before importing OmniGibson')
    version = importlib.metadata.version('isaacsim')
    if version != '6.0.1.0':
        raise RuntimeError(f'Compatibility requires Isaac Sim 6.0.1.0, got {version}')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in manifest['kit_files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Kit digest mismatch: {name}')
    sys.meta_path.insert(0, _Finder(manifest['modules']))
    record = dict(manifest, engine=version, mode='external-in-memory-source-adaptation',
                  official_runtime=False, upstream_files_modified=False,
                  renderer_override=os.environ.get('WORLD_BEHAVIOR_RENDERER', 'upstream'),
                  render_trace=os.environ.get('WORLD_BEHAVIOR_RENDER_TRACE'),
                  local_adapter_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in ('__init__.py', 'render_diagnostics.py')})
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'compatibility.json').write_text(json.dumps(record, indent=2))
