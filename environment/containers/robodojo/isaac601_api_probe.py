"""Run with Isaac Sim 6.0.1 Python; no environment or module patches.

Use a script file rather than embedding multiline Python in Kit's argv.
Example: python isaac601_api_probe.py --output /runs/api-check.json --allow-root
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args, kit_args = parser.parse_known_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("isaacsim")
    if version != "6.0.1.0":
        raise RuntimeError(f"Expected isaacsim 6.0.1.0, got {version}")
    sys.argv = [sys.argv[0], *kit_args]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    result = {"isaacsim": version, "scene_loaded": False, "robot_moved": False, "checks": {}}
    try:
        app.update()
        for name, code in {
            "robodojo_sim_import": "from isaaclab.sim import PhysxCfg, SimulationCfg",
            "robodojo_config_import": "from isaaclab.utils import configclass",
            "robodojo_env_import": "from isaaclab.envs import DirectRLEnvCfg",
        }.items():
            try:
                exec(code, {})
                result["checks"][name] = {"ok": True}
            except Exception:
                result["checks"][name] = {"ok": False, "error": traceback.format_exc()}
        # Kit fast shutdown may exit the process; persist before app.close().
        output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        app.close()


if __name__ == "__main__":
    main()
