"""Restore the pinned A1's explicit zero friction across legacy/new PhysX APIs.

IsaacLab 2.2's >=5 friction writer only edits the new API's returned tensor.
On legacy USD joints this leaves the old load-dependent coefficient active.
This adapter supports ONLY the pinned all-zero A1 contract: nonzero legacy
coefficients cannot be translated to new friction efforts by copying numbers.
"""
import json
from pathlib import Path


def enforce_zero_friction(robot):
    import torch

    if len(robot.joint_names) != 12 or set(robot.actuators) != {'legs'}:
        raise RuntimeError('Unexpected A1 actuator contract')
    actuator = robot.actuators['legs']
    if actuator.cfg.friction != 0.0 or torch.any(actuator.friction != 0):
        raise RuntimeError('A1 friction compatibility only supports explicit upstream friction=0')
    view = robot.root_physx_view
    ids = torch.arange(robot.num_instances, dtype=torch.int32, device='cpu')
    legacy_before = view.get_dof_friction_coefficients().clone()
    new_before = view.get_dof_friction_properties().clone()
    if torch.any(new_before != 0):
        raise RuntimeError('Unexpected nonzero modern friction; refusing to redefine the task')
    view.set_dof_friction_coefficients(torch.zeros_like(legacy_before), ids)
    view.set_dof_friction_properties(torch.zeros_like(new_before), ids)
    legacy_after = view.get_dof_friction_coefficients().clone()
    new_after = view.get_dof_friction_properties().clone()
    if torch.any(legacy_after != 0) or torch.any(new_after != 0):
        raise RuntimeError('PhysX did not apply the pinned zero joint friction')
    return {'upstream_requested_friction': 0.0,
            'legacy_before': legacy_before.tolist(), 'legacy_after': legacy_after.tolist(),
            'new_before': new_before.tolist(), 'new_after': new_after.tolist(),
            'readback_verified': True, 'upstream_files_modified': False,
            'reason': 'Honor explicit A1 zero friction in both PhysX representations; not fitted to trajectory'}


def articulation_class(output):
    from isaaclab.assets import Articulation

    class A1ZeroFrictionArticulation(Articulation):
        def _process_actuators_cfg(self):
            super()._process_actuators_cfg()
            audit = enforce_zero_friction(self)
            Path(output, 'a1-friction-compatibility.json').write_text(json.dumps(audit, indent=2) + '\n')

    return A1ZeroFrictionArticulation
