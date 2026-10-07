"""Start only upstream SimulationApp: no RoboLab, assets, or compatibility layer.

Run inside the selected image using its bundled python.sh. Evidence goes to
/runs, which must be a writable bind mount. This is not an evaluation episode.
"""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

out = Path('/runs')
versions = {}
for package in ('isaacsim', 'isaaclab', 'torch', 'warp-lang'):
    try:
        versions[package] = version(package)
    except PackageNotFoundError:
        versions[package] = None
for path in ('/workspace/isaaclab/VERSION', '/isaac-sim/VERSION'):
    if Path(path).is_file():
        versions[path] = Path(path).read_text().strip()
(out / 'versions.json').write_text(json.dumps(versions, indent=2))
(out / 'stage.txt').write_text('before SimulationApp import\n')
try:
    from isaacsim import SimulationApp
    (out / 'stage.txt').write_text('before SimulationApp construction\n')
    app = SimulationApp({'headless': True})
    try:
        (out / 'stage.txt').write_text('SimulationApp constructed\n')
        for _ in range(3):
            app.update()
        (out / 'sim-probe.json').write_text(json.dumps({
            'updates': 3, 'probe_only': True, 'upstream_compatibility_loaded': False,
        }, indent=2))
    finally:
        app.close()
except BaseException:
    import traceback
    (out / 'error.txt').write_text(traceback.format_exc())
    raise
