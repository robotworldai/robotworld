"""One observation camera; every control step is encoded, independent of prompt history."""
import subprocess
from pathlib import Path
import numpy as np

class Camera:
    def __init__(self,sim,output):
        import omni.replicator.core as rep
        self.sim=sim;self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.camera=rep.create.camera(position=(-1.8,1.9,1.65),look_at=(-.2,0,.8),focal_length=20)
        self.product=rep.create.render_product(self.camera,(640,480))
        self.rgb=rep.AnnotatorRegistry.get_annotator('rgb');self.rgb.attach(self.product)
        self.frames=0
        for _ in range(20):self.render()
        self.log=(self.out/'camera.ffmpeg.log').open('w')
        self.writer=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo','-pixel_format','rgb24','-video_size','640x480','-framerate',f'{1/(sim.world.get_physics_dt()*2):.6f}','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(self.out/'camera.mp4')],stdin=subprocess.PIPE,stderr=self.log)
    def render(self):
        import carb,omni.kit.app
        settings=carb.settings.get_settings();key='/app/player/playSimulations';old=settings.get(key)
        before=self.sim.world.current_time
        settings.set(key,False)
        try:
            import omni.replicator.core as rep
            rep.orchestrator.step(delta_time=0.0,pause_timeline=False)
        finally:settings.set(key,old)
        if self.sim.world.current_time!=before:raise RuntimeError('Rendering unexpectedly advanced physics time')
        a=np.asarray(self.rgb.get_data())
        if a.shape!=(480,640,4):return None
        self.image=np.ascontiguousarray(a[:,:,:3]);return self.image
    def capture(self):
        for _ in range(3):
            image=self.render()
            if image is not None:break
        if image is None:raise RuntimeError('Camera produced no RGB data: '+str(np.asarray(self.rgb.get_data()).shape))
        self.writer.stdin.write(image.tobytes());self.frames+=1
        return image
    def close(self):
        self.writer.stdin.close();code=self.writer.wait(timeout=60);self.log.close()
        if code:raise RuntimeError('Video encoding failed')
