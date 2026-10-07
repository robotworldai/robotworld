from __future__ import annotations


from pxr import UsdGeom, Gf
DEFAULT_CAMERA = ["/OmniverseKit_Persp"]
OUTPUT_ROOT = "_rgb_recordings"  # Base output folder (relative to CWD)
RESOLUTION = (1920, 1080)  # (width, height) for render products
NUM_FRAMES = 300  # Total frames to capture (≈10s at 30 steps/s)
FRAME_PADDING = 6  # Zero-padding for filenames (e.g., 000123)
RENDER_PRODUCT_NAME_FMT = "cam_{idx:02d}"  # Folder name per camera under output

import os, time
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.replicator.core")
enable_extension("omni.replicator.isaac")

import omni.replicator.core as rep
import carb


class FrameRecorder:
    def __init__(
        self,
        camera_prim_paths=DEFAULT_CAMERA,
        output_root=OUTPUT_ROOT,
        resolution=RESOLUTION,
        frame_padding=FRAME_PADDING,
    ):
        temp_cam_prim_path = list(camera_prim_paths)
        self.camera_prim_paths = temp_cam_prim_path
        self.output_root = output_root
        self.resolution = resolution
        self.frame_padding = frame_padding
        self._writer = None
        self._rps = []
        self._out_dir = None

    @property
    def output_dir(self):
        return self._out_dir

    def _ensure_stage(self):
        if omni.usd.get_context().get_stage() is None:
            raise RuntimeError("[LiveRGB] No USD stage loaded.")

    def _validate_cams(self):
        stage = omni.usd.get_context().get_stage()
        ok = []
        for p in self.camera_prim_paths:
            prim = stage.GetPrimAtPath(p)
            if not prim.IsValid():
                carb.log_warn(f"[LiveRGB] Skipping non-existent prim: {p}")
                continue
            ok.append(p)
        if not ok:
            raise ValueError("[LiveRGB] No valid camera prims.")
        self.camera_prim_paths = ok

    def _make_out_dir(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._out_dir = os.path.abspath(
            os.path.join(os.getcwd(), self.output_root, f"run_{ts}")
        )
        os.makedirs(self._out_dir, exist_ok=True)

    def start(self):
        self._ensure_stage()
        self._validate_cams()
        self._make_out_dir()

        # render products
        self._rps = []
        for i, p in enumerate(self.camera_prim_paths):
            rp = rep.create.render_product(p, self.resolution, name=f"cam_{i:02d}")
            self._rps.append(rp)

        # writer: RGB-only PNGs
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(
            output_dir=self._out_dir,
            rgb=True,
            image_output_format="png",
            frame_padding=self.frame_padding,
        )
        w.attach(self._rps)
        self._writer = w
        carb.log_info(f"[LiveRGB] Recording to: {self._out_dir}")

    def capture(self):
        pass

    def stop(self):
        if self._writer:
            self._writer.detach()
        for rp in self._rps:
            rp.destroy()
        rep.orchestrator.wait_until_complete()
        carb.log_info(f"[LiveRGB] Done: {self._out_dir}")
        return self._out_dir


def ensure_world_camera(
    path="/World/RecorderCam",
    pos=Gf.Vec3d(3.0, 0.0, 1.2),
    euler_deg=Gf.Vec3f(-10.0, 0.0, 180.0),
):
    import omni.usd, omni.kit.commands

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_path=path,
            prim_type="Camera",
            select_new_prim=False,
        )
    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(pos)
    xform.SetRotate(euler_deg)
    return path
