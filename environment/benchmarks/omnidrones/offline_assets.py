"""Map only the upstream default ground to byte-verified Isaac 4.1 assets."""
import functools
import hashlib
import json
from pathlib import Path

FILES = {
    'default_environment.usd': 'e6d9de28cebb4d646f6f311502d4bc2ae2a5a3a96a1006334b562b214eba39c4',
    'Materials/Textures/Wireframe_blue.png': '940b96ee8c8e53bd1acbc770d4ca0ac98bed9c96439ab386d1b27dbc9790cc4e',
    'Materials/Textures/WireframeBlur_basecolor.png': 'deab39088eaf39ed7be951966c5e6bd5e58b90ab64743b0cdc8e5579fd5a99d9',
    'Materials/Textures/WireframeBlur_blue.png': '44fc0e82007bb0bd94631cd7ec45986a71ed74b9413899fbffac13efe151be4b',
}


def install(kit_utils, root, report):
    root = Path(root)
    for name, digest in FILES.items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Missing/modified original Isaac4.1 ground asset: '+str(path))
    original = kit_utils.create_ground_plane

    @functools.wraps(original)
    def create_ground_plane(*args, **kwargs):
        if 'usd_path' not in kwargs:
            kwargs['usd_path'] = str(root/'default_environment.usd')
        return original(*args, **kwargs)

    kit_utils.create_ground_plane = create_ground_plane
    Path(report).write_text(json.dumps(dict(isaac_version='4.1', root=str(root),
        files=FILES, upstream_usd_unchanged=True, physics_parameters_unchanged=True), indent=2))
