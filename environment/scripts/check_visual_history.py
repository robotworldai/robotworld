"""Bounded real Codex/Astra image-window probe, with no simulator or task assets."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import secrets
import tomllib

from PIL import Image, ImageDraw

from environment.benchmarks.behavior_1k.run import model_overrides
from environment.runtime.episode_runner import run_episode
from environment.runtime.image_history import ObservationHistory, INSTRUCTIONS
from environment.runtime.request_audit import RequestAudit


class Probe:
    require_isolated_thread = True
    read_only_tools = {"observe"}

    def __init__(self, output, rounds):
        self.output, self.rounds = output, rounds
        self.calls = 0
        self.answers = []
        self.history = ObservationHistory()
        self.current = None
        self.code = None
        self.instructions = (
            f"Synthetic visual transport test, not a robot task. Call observe exactly {rounds} times sequentially. "
            "On EVERY call pass previous_code equal to the four characters in the latest CURRENT image. "
            "Read the image itself; earlier HISTORY images have different codes. "
            f"After call {rounds}, output FINAL_CODE= followed by the code in its CURRENT image. "
            "If an image is unavailable report failure, never guess. " + INSTRUCTIONS)
        self.developer_instructions = self.task_instruction = self.instructions

    def picture(self):
        self.code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
        picture = Image.new("RGB", (256, 296), "#eeeedd")
        ImageDraw.Draw(picture).text((12, 100), self.code, fill="black", font_size=70)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        self.current = buffer.getvalue()
        (self.output / f"frame-{self.calls:03d}.png").write_bytes(self.current)
        parts = self.history.append(self.calls, self.calls, [dict(
            type="inputImage", imageUrl="data:image/png;base64," + base64.b64encode(self.current).decode())])
        return [dict(type="inputText", text=f"Completed {self.calls}/{self.rounds} observe calls."), *parts]

    def observe_content(self):
        return self.picture()

    def observe(self, arguments):
        if self.calls >= self.rounds:
            raise RuntimeError("Synthetic probe observation budget exceeded")
        correct = arguments.get("previous_code", "").strip().upper() == self.code
        self.answers.append(correct)
        self.calls += 1
        return dict(success=True, contentItems=self.picture()), dict(synthetic=True, previous_code_correct=correct)

    def tool_specs(self):
        return [dict(type="function", name="observe", description="Read a new synthetic image.",
                     inputSchema=dict(type="object", properties={"previous_code": dict(type="string")},
                                      required=["previous_code"], additionalProperties=False))]

    def tool_handlers(self):
        return {"observe": self.observe}

    def ended(self):
        return False

    def official_success(self):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=52)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 51 <= args.rounds <= 60:
        parser.error("Use 51..60 rounds to cross the old image limit with bounded cost")
    if not 900 <= args.timeout_seconds <= 1800:
        parser.error("Probe timeout must be between 900 and 1800 seconds")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = tomllib.loads((Path.home() / ".codex/config.toml").read_text())
    probe = Probe(args.output, args.rounds)
    with RequestAudit(cfg, args.output / "request-audit.json", {"observe"},
                      current_image=lambda: probe.current, image_window=probe.history.snapshot,
                      aihub_direct=True, max_requests=args.rounds+4, max_retries=3) as audit:
        result = run_episode(probe, manifest=args.manifest, output_dir=args.output, model=cfg["model"],
                             config_overrides=model_overrides() + audit.overrides,
                             max_actions=1, max_tools=args.rounds+1, timeout_s=args.timeout_seconds, allow_give_up=False,
                             on_interrupt=audit.controller_interrupt)
    answer = " ".join(result.get("agent_messages", [])).replace(" ", "").upper()
    checks = dict(audit_valid=audit.valid, expected_calls=probe.calls == args.rounds,
                  image_answers_correct=bool(probe.answers) and all(probe.answers),
                  final_image_correct=f"FINAL_CODE={probe.code}" in answer,
                  completed=result["termination"] == "agent_completed",
                  raw_images_crossed_50=any(r.get("input_images", 0) > 50 for r in audit.records),
                  outgoing_at_most_five=bool(audit.records) and all(0 < r.get("outgoing_input_images", 0) <= 5 for r in audit.records))
    report = dict(passed=all(checks.values()), checks=checks, calls=probe.calls, requests=len(audit.records),
                  correct_image_answers=sum(probe.answers), target_calls=args.rounds,
                  timeout_seconds=args.timeout_seconds,
                  termination=result["termination"], model=cfg["model"], image_history=probe.history.snapshot()["policy"],
                  error=result.get("error"), simulator_started=False)
    (args.output / "probe-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
