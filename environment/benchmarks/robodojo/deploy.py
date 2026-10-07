"""External episode entry point. Does not install files into RoboDojo."""
import json
import os
from pathlib import Path
import tomllib

from environment.runtime.episode_runner import run_episode
from environment.runtime.request_audit import RequestAudit
from .adapter import RoboDojoAdapter
from environment.runtime.nonaction_budget import enabled


def eval_one_episode(TASK_ENV, model_client=None, *, manifest, output_dir,
                     model=None, instruction=None, max_actions=8,
                     timeout_s=300, config_overrides=(), decode_image=None):
    del model_client
    adapter = RoboDojoAdapter(TASK_ENV, output_dir=output_dir, decode_image=decode_image,
                             max_actions=max_actions, record_video=True)
    try:
        if os.environ.get('WORLD_CODEX_SOCKET'):
            from environment.runtime.image_history import publish_image_window
            observe = adapter.observe_content
            def published_observation():
                parts = observe()
                publish_image_window(output_dir, parts, adapter.visual_history.included)
                return parts
            adapter.observe_content = published_observation
            result = run_episode(adapter, manifest=manifest, output_dir=output_dir,
                                 model=model, instruction=instruction, max_actions=max_actions,
                                 timeout_s=timeout_s, config_overrides=config_overrides,
                                 recover_failed_turns=True, allow_give_up=False,
                                 continue_on_completion=enabled(), nonaction_protocol=enabled())
            result.update(image_history=adapter.visual_history.snapshot()['policy'],
                          request_boundary_audit='host proxy.json; not independently asserted here')
            (Path(output_dir)/'episode.json').write_text(json.dumps(result,indent=2)+'\n')
            return result
        config = tomllib.loads((Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml").read_text())
        with RequestAudit(config, Path(output_dir) / "request-audit.json", {"move_eef"},
                          image_window=adapter.visual_history.snapshot,
                          max_request_bytes=256 * 1024**2, max_requests=max_actions + 16) as audit:
            result = run_episode(adapter, manifest=manifest, output_dir=output_dir,
                                 model=model, instruction=instruction, max_actions=max_actions,
                                 timeout_s=timeout_s, config_overrides=[*config_overrides, *audit.overrides],
                                 on_interrupt=audit.controller_interrupt, recover_failed_turns=True,
                                 allow_give_up=False, continue_on_completion=enabled(),
                                 nonaction_protocol=enabled())
        result.update(image_history=adapter.visual_history.snapshot()["policy"], request_boundary_valid=audit.valid)
        (Path(output_dir) / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
        if not audit.valid:
            raise RuntimeError("RoboDojo model request/image boundary failed; result is not a valid score")
        return result
    finally:
        if adapter.video:
            adapter.video.close()
