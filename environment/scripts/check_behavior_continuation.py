"""Two-turn synthetic integration test for long-budget Codex continuation."""
import argparse
import json
import os
from pathlib import Path
import tomllib

from environment.scripts.check_behavior_images import Probe
from environment.benchmarks.behavior_1k.run import model_overrides
from environment.runtime.episode_runner import run_episode
from environment.runtime.request_audit import RequestAudit


class ContinuationProbe(Probe):
    instructions = (
        "Synthetic continuation test, not a robot task. On your FIRST turn call step exactly once, "
        "then finish with the text PAUSE. Do not call it twice on that first turn. "
        "If a later user message asks you to continue, call step once more. "
        "step accepts an empty object. No other tools are available.")
    task_instruction = instructions
    developer_instructions = instructions
    read_only_tools = set()

    def __init__(self):
        super().__init__()
        self.state = dict(native_steps=0)

    def tool_specs(self):
        return [dict(type="function", name="step", description="Advance synthetic counter once.",
                     inputSchema=dict(type="object", properties={}, additionalProperties=False))]

    def tool_handlers(self):
        return {"step": self.step}

    def step(self, arguments):
        if arguments:
            raise ValueError("Expected empty arguments")
        self.state["native_steps"] += 1
        return dict(success=True, contentItems=self.picture(True)), dict(executed_native_steps=1)

    def ended(self):
        return self.state["native_steps"] >= 2

    def official_success(self):
        return self.ended()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tomllib.loads((Path.home() / ".codex/config.toml").read_text())
    probe = ContinuationProbe()
    with RequestAudit(cfg, args.output / "request-audit.json", {"step"},
                      current_image=lambda: probe.current, max_requests=6,
                      aihub_direct=True, max_retries=3) as audit:
        result = run_episode(probe, manifest=args.manifest, output_dir=args.output,
                             model=cfg["model"], config_overrides=model_overrides() + audit.overrides,
                             max_actions=2, max_tools=2, timeout_s=600, allow_give_up=False,
                             continue_on_completion=True)
    passed = bool(audit.valid and result["action_calls"] == 2 and
                  len(result["continuations"]) == 1 and result["termination"] == "environment_end")
    report = dict(passed=passed, request_boundary_valid=audit.valid,
                  continuations=len(result["continuations"]), termination=result["termination"])
    (args.output / "probe-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
