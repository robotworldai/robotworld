"""External official-API launcher: never writes a policy into RoboDojo.

Launch with RoboDojo's compatible Isaac Python and PYTHONPATH containing World,
the RoboDojo root and its XPolicyLab directory. Isaac 6 compatibility is opt-in.
"""
import argparse
import asyncio
import json
import os
import inspect
from pathlib import Path
import threading
import traceback
from datetime import datetime, timezone


SMOKE = """This is a move_eef interface smoke test, not a task success attempt.
Use the provided measured state. Perform these steps in order, one move_eef per step:
1. Command left_z to its initial observed value plus 0.01 metres; keep all other dimensions unchanged.
2. Return left_z to the initial value.
3. Command right_gripper to 1.0 (open).
4. Send left_z=99 once to verify rejection. After rejection do not retry that target.
Then finish with a short report. Never claim task success. If a movement is unreachable,
report it and continue only with the non-moving gripper/validation checks.
"""


class ResetOnlyModel:
    def reset(self):
        pass

    def get_action(self):
        raise RuntimeError("Actions must come from Codex dynamic tools, not this compatibility endpoint")


def main():
    # Resolve the official control protocol dependency before Kit adds its
    # bundled libraries to sys.path. No upstream modules are patched.
    import websockets.asyncio.server

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-catalog", type=Path)
    parser.add_argument("--task", default="general_pickup")
    parser.add_argument("--layout", type=int, default=0, help="Official SeedManager index, not necessarily filename suffix")
    parser.add_argument("--eval-seed", type=int, default=0, help="Eval_Layout/<config>/<seed> directory")
    parser.add_argument("--config", default="arx_x5")
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true", help="Initialize official simulator and capture observation; no model call")
    parser.add_argument("--probe-rigid-binding", action="store_true")
    parser.add_argument('--probe-startup-analysis',choices=('contacts','coin-exact-mesh','socket-singleton','socket-binding'))
    parser.add_argument("--isaac601-compat", action="store_true", help="Enable the explicitly authorized external Isaac 6.0.1 compatibility layer")
    parser.add_argument("--run-task", action="store_true", help="Use the official task instruction instead of the four-action interface smoke")
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--max-actions", type=int, default=32, help="Tool-call budget for --run-task")
    parser.add_argument("--timeout", type=float, default=300)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.probe_startup_analysis and (not args.probe_only or args.task !=
            ('deposit_coin' if args.probe_startup_analysis in ('contacts','coin-exact-mesh') else 'plug_in_charger')):
        parser.error('Startup analysis is restricted to its unscored diagnostic task')
    if args.probe_rigid_binding and (not args.probe_only or args.task not in ('pour_liquid_into_cup','pour_by_language')):
        parser.error('Rigid binding candidate is only allowed for the pour diagnostic')
    args.root = args.root.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()
    if args.model_catalog is not None:
        args.model_catalog = args.model_catalog.resolve()
    if not args.probe_only and args.manifest is None:
        parser.error("--manifest is required for Codex execution")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    def progress(phase):
        (output / "progress.json").write_text(json.dumps({
            "phase": phase, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        print(f"[World] {phase}", flush=True)

    progress("initializing_simulator")
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    # Official eval paths are relative; contain output in our run directory.
    os.chdir(output)
    if args.isaac601_compat:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False,
                             "width": 640, "height": 480,
                             "extra_args": ["--/app/updateOrder/checkForHydraRenderComplete=1000",
                                            "--/app/renderer/waitIdle=true", "--/app/hydraEngine/waitIdle=true"]})
        (output / "render-launch.json").write_text(json.dumps({
            "zero_delay_settings": True,
            "source": "RoboDojo env/camera_manager/capture/render_sync.py ZERO_DELAY_KIT_SETTINGS",
            "physics_parameters_changed": False}, indent=2))
    else:
        app = AppLauncher(args).app
    env = None
    server = None
    loop = None
    exit_code = 0
    try:
        if args.isaac601_compat:
            progress("installing_external_compatibility")
            from environment.benchmarks.robodojo.compat.isaac601 import install
            install(app)
            if args.task=='deposit_coin' and not args.probe_startup_analysis:
                from environment.benchmarks.robodojo.compat.coin_stand import install as install_coin_stand
                install_coin_stand(output)
            if args.probe_rigid_binding:
                from environment.validation.robodojo_rigid_binding import install as install_binding_candidate
                install_binding_candidate(output,category='wine_bottle' if args.task=='pour_by_language' else 'wuliangye')
            elif not args.probe_only and args.task in ('pour_liquid_into_cup','pour_by_language'):
                from environment.validation.robodojo_rigid_binding import install as install_verified_binding
                install_verified_binding(output,verified=True,category='wine_bottle' if args.task=='pour_by_language' else 'wuliangye')
            (output / "compatibility.json").write_text(json.dumps({
                "isaacsim": "6.0.1.0", "external_runtime_compatibility": True,
                "upstream_source_modified": False}, indent=2))
            if args.task == "match_and_pick_from_conveyor":
                import omni.kit.app
                extensions = omni.kit.app.get_app().get_extension_manager()
                extensions.set_extension_enabled_immediate("isaacsim.asset.gen.conveyor", True)
                app.update()
                if not extensions.is_extension_enabled("isaacsim.asset.gen.conveyor"):
                    raise RuntimeError("Conveyor extension could not be enabled")
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client.eval_env import create_eval_env
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        from XPolicyLab.utils.process_data import decode_image_bit
        from task.RoboDojo import task_registry
        from environment.benchmarks.robodojo.adapter import RoboDojoAdapter
        from environment.benchmarks.robodojo.deploy import eval_one_episode

        # EvalEnv publicly requires a reset/handshake server even though it is not
        # used for policy inference. Use the official protocol, no module patching.
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        server = PolicyServer(ResetOnlyModel(), PolicyServerConfig(host="127.0.0.1", port=0))
        asyncio.run_coroutine_threadsafe(server.start(), loop).result(20)
        port = int(server.url.rsplit(":", 1)[1])
        config_dir = args.root.resolve() / "env_cfg"
        task, _ = task_registry.load_task_class(args.task)
        base = load_yaml(str(config_dir / f"{args.config}.yml"))
        base.update(task_name=task, num_envs=1, seed=args.eval_seed, eval_num=1, eval_batch=False,
                    policy_name="WorldCodex", additional_info="move-eef-smoke")
        config = OmegaConf.create({
            **{key: load_yaml(str(config_dir / key / f"{base['config'][key]}.yml"))
               for key in ("sim", "scene", "camera", "robot")},
            "task_env": load_yaml(task_registry.task_config_path(str(args.root.resolve() / "task/RoboDojo/config"), task)),
            "eval_cfg": base, "deploy_cfg": {"port": port, "host": "127.0.0.1"}})
        config = process_randomization(config)
        config, _ = process_config(config, task_name=task)
        OmegaConf.update(config, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(config, "camera.default_frequency", 25, force_add=True)
        config.sim.seed = [0]
        env = create_eval_env(config, app)
        progress("resetting_official_environment")
        if args.isaac601_compat and args.task=='plug_in_charger' and not args.probe_only:
            from environment.validation.robodojo_contact_probe import startup_analysis
            with startup_analysis(output,'socket-binding',verified_socket=True):
                env.reset(seed=[args.layout])
        elif args.probe_only and args.task == 'plug_in_charger':
            from environment.validation.robodojo_startup import trace_initial_physics, trace_reset_motion, trace_stability
            from contextlib import nullcontext
            from environment.validation.robodojo_contact_probe import startup_analysis
            with (startup_analysis(output,args.probe_startup_analysis) if args.probe_startup_analysis else nullcontext()), trace_initial_physics(output), trace_reset_motion(env,output), trace_stability(env,output):
                env.reset(seed=[args.layout])
        elif args.probe_only and args.task in ('pour_liquid_into_cup','pour_by_language','deposit_coin'):
            from environment.validation.robodojo_startup import trace_stability, trace_reset_motion
            from contextlib import nullcontext
            from environment.validation.robodojo_contact_probe import startup_analysis
            with (startup_analysis(output,args.probe_startup_analysis) if args.probe_startup_analysis else nullcontext()), trace_reset_motion(env,output), trace_stability(env,output):
                env.reset(seed=[args.layout])
        else:
            env.reset(seed=[args.layout])
        from environment.benchmarks.robodojo.lifecycle import prepare_episode
        prepare_episode(env)
        original_step_limit = int(env.step_lim)
        if args.max_env_steps is not None:
            if args.max_env_steps <= 0:
                raise ValueError("max-env-steps must be positive")
            env.step_lim = args.max_env_steps
        (output / "evaluation-config.json").write_text(json.dumps({
            "task": args.task, "eval_seed": args.eval_seed, "layout": args.layout,
            "original_step_limit": original_step_limit, "effective_step_limit": int(env.step_lim),
            "standard_step_budget": int(env.step_lim) == original_step_limit,
            "max_actions": args.max_actions, "model_timeout": args.timeout}, indent=2))
        if args.isaac601_compat and args.task == "match_and_pick_from_conveyor":
            from environment.benchmarks.robodojo.compat.conveyor import snapshot, repair_conveyor_variable_targets
            from environment.benchmarks.robodojo.compat.isaac601 import update_without_physics
            import carb
            before = snapshot()
            conveyors = [o for o in env.scene_manager.get_objects(env_ids=[0], object_type="rigid").values()
                         if o.instance_config.get("label") == "conveyor"]
            if len(conveyors) != 1:
                raise RuntimeError("Expected one conveyor for the external compatibility repair")
            repairs = repair_conveyor_variable_targets(conveyors[0].usd_path)
            update_without_physics(app, carb.settings.get_settings())
            (output / "conveyor-compatibility.json").write_text(json.dumps({
                "before": before, "repairs": repairs, "after": snapshot()}, indent=2))
            from environment.benchmarks.robodojo.diagnostics import scene_snapshot
            scene_initial = scene_snapshot(env)
            (output / "scene-before.json").write_text(json.dumps(scene_initial, indent=2))
            import numpy as np
            for repair in repairs:
                override = repair.get("surface_override")
                if override:
                    if len(scene_initial["surfaces"]) != 1 or not np.allclose(
                            scene_initial["surfaces"][0]["authored_vector"],
                            override["authored_surface_velocity"], atol=1e-6):
                        raise RuntimeError("Conveyor graph did not preserve the source physical velocity")
        if args.probe_only:
            progress("capturing_observation")
            adapter = RoboDojoAdapter(env, output_dir=output, decode_image=decode_image_bit,
                                     record_video=bool(args.diagnostic_steps))
            if args.diagnostic_steps:
                from environment.benchmarks.robodojo.diagnostics import run_hold_diagnostic
                run_hold_diagnostic(env, adapter, args.diagnostic_steps, output)
            content = adapter.observe_content()
            (output / "probe.json").write_text(json.dumps({"status": "initialized",
                "observation": [x for x in content if x["type"] == "inputText"]}, indent=2))
        else:
            progress("running_source_codex")
            result = eval_one_episode(env, manifest=args.manifest, output_dir=output,
                             model=args.model, instruction=None if args.run_task else SMOKE,
                             max_actions=args.max_actions if args.run_task else 4,
                             timeout_s=args.timeout, decode_image=decode_image_bit,
                             config_overrides=([f'model_catalog_json={json.dumps(str(args.model_catalog))}']
                                               if args.model_catalog else []))
            if args.run_task:
                from environment.benchmarks.robodojo.lifecycle import finish_task_episode
                result["official_success"] = finish_task_episode(env)
                result["official_episode_finalized"] = True
                (output / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.isaac601_compat and args.task == "match_and_pick_from_conveyor":
            (output / "scene-after.json").write_text(json.dumps(scene_snapshot(env), indent=2))
        progress("episode_output_saved")
    except BaseException as error:
        exit_code = 1
        progress("failed")
        (output / "failure.json").write_text(json.dumps({"error_type": type(error).__name__,
                                                        "error": str(error)}, indent=2))
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            try:
                env.model_client.close()
            finally:
                env.close()
        if server is not None:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result(20)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        # Isaac 6 may call os._exit inside close. Save exceptions first and
        # propagate its public exit_code argument where supported.
        close_args = {"exit_code": exit_code} if "exit_code" in inspect.signature(app.close).parameters else {}
        app.close(**close_args)


if __name__ == "__main__":
    main()
