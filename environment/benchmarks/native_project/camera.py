"""Kit render-product pump for native task review, never a policy sensor."""
import numpy as np
from environment.benchmarks.wheeledlab.camera import Camera as BaseCamera


class Camera(BaseCamera):
    def __init__(self,sim,output):
        from .review_scene import ReviewBackdrop
        pos=sim.env.scene['robot'].data.root_pos_w[0].detach().cpu().tolist()
        self.backdrop=ReviewBackdrop(output,pos)
        super().__init__(sim,output)

    def render(self, overview=False):
        import carb
        import omni.replicator.core as rep
        from pxr import Gf
        pos = self.sim.env.scene['robot'].data.root_pos_w[0].detach().cpu().tolist()
        from .review_scene import camera_offsets
        offset, focus = camera_offsets(self.sim.case)
        eye = self.sim.env.cfg.viewer.eye if overview else tuple(pos[i]+offset[i] for i in range(3))
        target = self.sim.env.cfg.viewer.lookat if overview else tuple(pos[i]+focus[i] for i in range(3))
        self.transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
        settings = carb.settings.get_settings()
        key = '/app/player/playSimulations'
        previous = settings.get(key)
        before = self.sim.env.sim.current_time
        settings.set(key,False)
        try:
            with self.backdrop.visible():
                rep.orchestrator.step(delta_time=0.,pause_timeline=False)
        finally:
            settings.set(key,previous)
        if self.sim.env.sim.current_time != before:
            raise RuntimeError('Review rendering advanced physics')
        raw = np.asarray(self.rgb.get_data())
        if raw.shape != (480,640,4):
            return None
        self.image = np.ascontiguousarray(raw[:,:,:3])
        return self.image
