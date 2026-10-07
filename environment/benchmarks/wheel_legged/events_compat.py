"""Load unchanged v2.3.2 mass randomizer with its original minimum-mass clamp."""
import ast
import hashlib
from pathlib import Path
import sys
import types


def install(source: Path):
    from isaaclab.envs import mdp
    from isaaclab.envs.mdp import events
    path = Path(source) / 'compat/isaaclab232_events.py'
    data = path.read_bytes()
    expected = (path.parent / 'isaaclab232_events.sha256').read_text().split()[0]
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError('Pinned official IsaacLab v2.3.2 event source hash mismatch')
    # Execute exact original class and its original helpers, with the compatible
    # runtime's real torch/assets/managers imports. Other unrelated v2.3.2 events
    # are not installed or modified, and no parameters are discarded.
    tree = ast.parse(data, filename=str(path))
    names = {'randomize_rigid_body_mass','_randomize_prop_by_op','_validate_scale_range'}
    selected = [n for n in tree.body if getattr(n,'name',None) in names]
    if len(selected) != len(names):raise RuntimeError('Unexpected official event source')
    module = types.ModuleType('world_wheel_legged_native_mass232')
    module.__dict__.update(vars(events))
    module.__name__ = 'world_wheel_legged_native_mass232'
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    body = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(body),str(path),'exec'),module.__dict__)
    mdp.randomize_rigid_body_mass = module.randomize_rigid_body_mass
