"""Ideal vehicle-mounted sensors. World pose is used only to mount the camera."""
import subprocess
from pathlib import Path

PROFILE = 'robotworld-onboard-v2'
DESCRIPTION = 'front RGB camera + ideal joint encoders and body IMU gyro/tilt; no global pose, map or navigation oracle'


def telemetry(names, positions, velocities, angular_velocity, roll, pitch):
    """Allowlist physical measurements, never forward the simulator observation dict."""
    wheels = {name: float(velocities[i]) for i, name in enumerate(names) if name.endswith('_wheel_throttle')}
    steering = {name: float(positions[i]) for i, name in enumerate(names) if name.endswith('_wheel_steer')}
    return {'wheel_encoders_rad_s': wheels, 'steering_encoders_rad': steering,
            'imu': {'angular_velocity_body_rad_s': list(angular_velocity), 'roll_rad': float(roll), 'pitch_rad': float(pitch)},
            'sensor_model': 'ideal simulated encoders and IMU attitude/gyro; no added sensor noise'}


class OnboardCamera:
    def __init__(self, sim, output, facing='front'):
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        if facing not in ('front','rear'):raise ValueError(facing)
        self.sim = sim; self.facing=facing;self.stem='onboard' if facing=='front' else 'rear';self.out = Path(output); self.out.mkdir(parents=True, exist_ok=True)
        self.frames = 0; self.image = None; self.pixels = None
        self.lowres = self.out.parent / ('sensor-frames/'+facing+'-lowres'); self.lowres.mkdir(parents=True, exist_ok=True)
        stage = omni.usd.get_context().get_stage()
        camera_path='/World/RobotWorldOnboardCamera'+facing.title()
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(12.); camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.03, 100.))
        self.transform = UsdGeom.Xformable(camera).AddTransformOp()
        self.product = rep.create.render_product(camera_path, (640, 360))
        self.rgb = rep.AnnotatorRegistry.get_annotator('rgb'); self.rgb.attach(self.product)
        # Make the mandatory stopping location physically visible. This is paint,
        # not a collision change or a private coordinate sent to the agent.
        if facing=='front' and 'mandatory_stop' in sim.spec:
            from .sensor_markings import add_stop_marking
            add_stop_marking(stage, sim.spec['mandatory_stop']['center'])
        self.update_pose()
        for _ in range(20): self.render()
        self.log = (self.out / (self.stem+'.ffmpeg.log')).open('w')
        self.writer = subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo','-pixel_format','rgb24',
            '-video_size','640x360','-framerate',str(1/sim.dt),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-g','50','-movflags','+frag_keyframe+empty_moov',
            str(self.out/(self.stem+'.mp4'))], stdin=subprocess.PIPE, stderr=self.log)
        self.capture()

    def update_pose(self):
        from pxr import Gf
        robot = self.sim.env.scene['robot'].data
        pos = robot.root_pos_w[0].detach().cpu().tolist()
        q = robot.root_quat_w[0].detach().cpu().tolist()
        rotation = Gf.Rotation(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        direction=1 if self.facing=='front' else -1
        eye = Gf.Vec3d(*pos) + rotation.TransformDir(Gf.Vec3d(.22*direction, 0, .24))
        forward = rotation.TransformDir(Gf.Vec3d(direction, 0, -.16 if self.facing=='front' else -.70)); up = rotation.TransformDir(Gf.Vec3d(0, 0, 1))
        self.transform.Set(Gf.Matrix4d().SetLookAt(eye, eye+forward, up).GetInverse())

    def render(self):
        import carb
        import numpy as np
        import omni.replicator.core as rep
        key = '/app/player/playSimulations'; settings = carb.settings.get_settings(); previous = settings.get(key)
        before = self.sim.env.sim.current_time; settings.set(key, False)
        try: rep.orchestrator.step(delta_time=0., pause_timeline=False)
        finally: settings.set(key, previous)
        if self.sim.env.sim.current_time != before: raise RuntimeError('Onboard rendering advanced physics')
        raw = np.asarray(self.rgb.get_data())
        if raw.shape != (360,640,4): return None
        return np.ascontiguousarray(raw[:,:,:3])

    def capture(self,frame=None):
        from PIL import Image
        if frame is None:
            self.update_pose()
            for _ in range(3):
                frame = self.render()
                if frame is not None: break
        if frame is None: raise RuntimeError('No onboard RGB pixels')
        self.image = frame
        if self.facing=='front':self.sim.policy_image = frame
        small = Image.fromarray(frame).resize((48,27), Image.Resampling.BILINEAR)
        small.save(self.lowres/f'{self.sim.steps:06d}.png')
        self.pixels = {'type':'image_pixels','encoding':'RGB8 row-major interleaved','width':48,'height':27,
                       'control_step':self.sim.steps,'pixels':[v for pixel in small.getdata() for v in pixel]}
        if self.frames == 0: Image.fromarray(frame).save(self.out/(self.stem+'-initial.png'))
        self.writer.stdin.write(frame.tobytes()); self.frames += 1

    def close(self):
        self.writer.stdin.close(); code = self.writer.wait(timeout=60); self.log.close()
        if code: raise RuntimeError('Onboard video encoding failed')
