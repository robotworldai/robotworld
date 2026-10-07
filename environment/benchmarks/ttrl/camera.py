"""Whole-table review camera; never exposed to the policy or physics controls."""
from environment.benchmarks.native_project.camera import Camera as NativeCamera


class Camera(NativeCamera):
    def render(self, overview=False):
        import carb
        import numpy as np
        import omni.replicator.core as rep
        from pxr import Gf

        # Original table is centered at the environment origin, robot starts
        # behind x=-1.6. A fixed oblique frame covers both table ends and T1.
        origin = self.sim.env.scene.env_origins[0].detach().cpu().tolist()
        eye = [origin[i] + (-3.5, -3.6, 2.8)[i] for i in range(3)]
        target = [origin[i] + (-.3, 0., .7)[i] for i in range(3)]
        self.transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        settings = carb.settings.get_settings()
        key = '/app/player/playSimulations'
        previous = settings.get(key)
        before = self.sim.env.sim.current_time
        settings.set(key, False)
        try:
            with self.backdrop.visible():
                rep.orchestrator.step(delta_time=0., pause_timeline=False)
        finally:
            settings.set(key, previous)
        if self.sim.env.sim.current_time != before:
            raise RuntimeError('Review rendering advanced physics')
        raw = np.asarray(self.rgb.get_data())
        if raw.shape != (480, 640, 4):
            return None
        self.image = np.ascontiguousarray(raw[:, :, :3])
        return self.image
