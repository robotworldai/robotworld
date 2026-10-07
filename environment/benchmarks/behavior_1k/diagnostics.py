"""Reusable boundary and video validation helpers."""
import json
import subprocess
from pathlib import Path

def model_overrides():
    # Per-process restrictions only; existing provider and authentication remain untouched.
    features = ("shell_tool", "unified_exec", "view_image", "multi_agent", "multi_agent_v2",
                "apps", "plugins", "remote_plugin", "browser_use", "computer_use",
                "image_generation", "memories", "hooks", "skill_search", "code_mode",
                "code_mode_host", "unbounded_connection_retries", "goals",
                "default_mode_request_user_input")
    return [*(f"features.{name}=false" for name in features),
            "features.skip_host_skill_discovery=true", "mcp_servers={}",
            "project_doc_max_bytes=0", 'sandbox_mode="read-only"',
            "agents.enabled=false", "tools.experimental_request_user_input.enabled=false",
            "tools.update_plan.enabled=false",
            'web_search="disabled"', 'developer_instructions=""',
            'model_reasoning_effort="medium"']

def verify_video(directory, steps):
    capture = json.loads((directory / "capture.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if (steps <= 0 or manifest["frames_per_camera"] != steps or manifest["encoding_errors"] or
            manifest["env_steps"] != list(range(1, steps+1)) or manifest["fps"] != 30 or
            [f["native_step"] for f in capture["frames"]] != list(range(1, steps+1)) or
            any(f["physics_callbacks"] != 4 for f in capture["frames"])):
        raise RuntimeError("Video frame/physics cadence mismatch")
    streams = {}
    for name in ("head_640x480", "head", "left_wrist", "right_wrist"):
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json",
            str(directory / (name + ".mp4"))], text=True))["streams"][0]
        expected = (640, 480) if name == "head_640x480" else (128, 128)
        if (int(probe["nb_read_frames"]) != steps or probe["avg_frame_rate"] != "30/1" or
                (probe["width"], probe["height"]) != expected):
            raise RuntimeError("Encoded video differs from captured frames")
        streams[name] = probe
    return dict(streams=streams, frames=steps, duration_seconds=steps/30,
                continuous=True, extra_physics_steps=0)

