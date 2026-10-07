"""Review camera recorded each control step; never exposed as native sensor data."""
import subprocess
from pathlib import Path
import numpy as np

class Camera:
    def __init__(self,sim,output):
        import omni.replicator.core as rep
        self.sim=sim;self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True);self.frames=0
        import omni.usd
        from pxr import UsdGeom
        camera=UsdGeom.Camera.Define(omni.usd.get_context().get_stage(),'/World/WorldReviewCamera')
        camera.CreateFocalLengthAttr(20.);camera.CreateHorizontalApertureAttr(20.955)
        self.transform=UsdGeom.Xformable(camera).AddTransformOp()
        self.product=rep.create.render_product('/World/WorldReviewCamera',(640,480));self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach(self.product)
        for _ in range(20):self.render()
        if sim.case=='elevation' or hasattr(sim,'spec'):
            from PIL import Image
            for _ in range(3):overview=self.render(overview=True)
            if overview is not None:Image.fromarray(overview).save(self.out/'terrain-overview.png')
            for _ in range(3):self.render()
        self.log=(self.out/'camera.ffmpeg.log').open('w')
        self.writer=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo','-pixel_format','rgb24','-video_size','640x480','-framerate',str(1/sim.dt),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-g','50','-movflags','+frag_keyframe+empty_moov',str(self.out/'camera.mp4')],stdin=subprocess.PIPE,stderr=self.log)
    def update_pose(self,overview=False):
        pos=self.sim.env.scene['robot'].data.root_pos_w[0].detach().cpu().tolist()
        from pxr import Gf
        eye=self.sim.env.cfg.viewer.eye if overview else (pos[0]+2.5,pos[1]-2.5,pos[2]+2)
        target=self.sim.env.cfg.viewer.lookat if overview else pos
        self.transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
    def render(self,overview=False):
        import carb,omni.replicator.core as rep
        self.update_pose(overview)
        settings=carb.settings.get_settings();key='/app/player/playSimulations';previous=settings.get(key)
        before=self.sim.env.sim.current_time;settings.set(key,False)
        try:
            if hasattr(self.sim,'spec'):rep.orchestrator.step(delta_time=0.,pause_timeline=False)
            else:self.sim.env.sim.render()
        finally:settings.set(key,previous)
        if self.sim.env.sim.current_time!=before:raise RuntimeError('Review rendering advanced physics')
        raw=np.asarray(self.rgb.get_data())
        if raw.shape!=(480,640,4):return None
        self.image=np.ascontiguousarray(raw[:,:,:3]);return self.image
    def capture(self,frame=None):
        if frame is None:
            for _ in range(3):
                frame=self.render()
                if frame is not None:break
        if frame is None:raise RuntimeError('No review camera pixels: '+str(np.asarray(self.rgb.get_data()).shape))
        self.image=frame
        if self.frames==0:
            from PIL import Image
            Image.fromarray(frame).save(self.out/'initial.png')
        self.writer.stdin.write(frame.tobytes());self.frames+=1
    def close(self):
        self.writer.stdin.close();code=self.writer.wait(timeout=60);self.log.close()
        if code:raise RuntimeError('Video encoding failed')
