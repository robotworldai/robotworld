"""Verify the migration repair honors the task and fails on ignored writes."""
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from environment.benchmarks.robot_lab.friction_compat import enforce_zero_friction


@unittest.skipIf(torch is None, 'Run in the Isaac image with Torch')
class FrictionCompatibilityTest(unittest.TestCase):
    def robot(self, ignore_write=False):
        class View:
            def __init__(self):
                self.legacy = torch.full((1, 12), .2)
                self.modern = torch.zeros((1, 12, 3))
            def get_dof_friction_coefficients(self): return self.legacy.clone()
            def get_dof_friction_properties(self): return self.modern.clone()
            def set_dof_friction_coefficients(self, value, ids):
                if not ignore_write: self.legacy[ids.long()] = value[ids.long()]
            def set_dof_friction_properties(self, value, ids): self.modern[ids.long()] = value[ids.long()]
        return SimpleNamespace(joint_names=list(range(12)), num_instances=1, root_physx_view=View(),
                               actuators={'legs': SimpleNamespace(cfg=SimpleNamespace(friction=0.), friction=torch.zeros((1, 12)))})

    def test_restores_explicit_zero_despite_modern_api_already_zero(self):
        robot = self.robot()
        audit = enforce_zero_friction(robot)
        self.assertTrue(audit['readback_verified'])
        self.assertTrue(torch.all(robot.root_physx_view.legacy == 0))
        self.assertGreater(audit['legacy_before'][0][0], 0)

    def test_rejects_nonzero_contract_instead_of_converting_units(self):
        robot = self.robot(); robot.actuators['legs'].cfg.friction = .2
        with self.assertRaisesRegex(RuntimeError, 'explicit upstream'): enforce_zero_friction(robot)

    def test_rejects_ignored_setter(self):
        with self.assertRaisesRegex(RuntimeError, 'did not apply'): enforce_zero_friction(self.robot(ignore_write=True))

    def test_preserves_unexpected_modern_physics_by_refusing(self):
        robot = self.robot(); robot.root_physx_view.modern[0, 0, 1] = .1
        with self.assertRaisesRegex(RuntimeError, 'refusing'): enforce_zero_friction(robot)


if __name__ == '__main__': unittest.main()
