"""Bounded synthetic image round trip through the actual EEF request boundary."""
import argparse
import base64
import io
import json
import os
import secrets
from pathlib import Path
import tomllib

from PIL import Image, ImageDraw
from environment.benchmarks.behavior_1k.run import model_overrides
from environment.runtime.episode_runner import run_episode
from environment.runtime.request_audit import RequestAudit


class Probe:
    instructions = (
        "This is a synthetic image transport test, NOT a robot task. "
        "Call observe exactly FOUR times, sequentially. On calls 2, 3 and 4, pass "
        "previous_code equal to the four characters visible in the preceding tool image. "
        "On call 1 omit previous_code. After call 4, write FINAL_CODE= followed by the "
        "four characters visible in the last tool image. Do not use the initial image. "
        "Do not guess if the tool image is unavailable; report the omission.")
    task_instruction = instructions
    developer_instructions = instructions
    read_only_tools = {"observe"}
    require_isolated_thread = True

    def __init__(self):
        self.calls = 0
        self.current = None
        self.expected = None
        self.answers = []

    def tool_specs(self):
        return [dict(type="function", name="observe", description="Return a NEW synthetic image.",
                     inputSchema=dict(type="object", properties={"previous_code": {"type": "string"}}, additionalProperties=False))]

    def tool_handlers(self):
        return {"observe": self.observe}

    def picture(self, tool=False):
        picture = Image.new("RGB", (256, 296), "#20e030" if tool else "#e03020")
        code = "".join(secrets.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(4)) if tool else "INIT"
        if tool:
            self.expected = code
        ImageDraw.Draw(picture).text((12, 100), code, fill="black", font_size=70)
        buf = io.BytesIO()
        picture.save(buf, format="PNG")
        self.current = buf.getvalue()
        return [dict(type="inputText", text="Synthetic test image; inspect its visual contents."),
                dict(type="inputImage", imageUrl="data:image/png;base64," + base64.b64encode(self.current).decode())]

    def observe_content(self):
        return self.picture()

    def observe(self, arguments):
        if self.calls:
            self.answers.append(arguments.get("previous_code", "").strip().upper() == self.expected)
        self.calls += 1
        if self.calls > 4:
            raise RuntimeError("Synthetic probe permits four observations only")
        return dict(success=True, contentItems=self.picture(True)), {"synthetic": True}

    def ended(self):
        return False

    def official_success(self):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aihub-direct", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(exist_ok=False, parents=True)
    cfg = tomllib.loads((Path.home() / ".codex/config.toml").read_text())
    probe = Probe()
    with RequestAudit(cfg, args.output / "request-audit.json", {"observe", "give_up"},
                      current_image=lambda: probe.current, max_requests=6,
                      aihub_direct=args.aihub_direct) as audit:
        result = run_episode(probe, manifest=args.manifest, output_dir=args.output,
                             model=cfg["model"], config_overrides=model_overrides() + audit.overrides,
                             max_actions=1, max_tools=5, timeout_s=180)
    answer = " ".join(result.get("agent_messages", [])).upper()
    passed = (audit.valid and probe.calls == 4 and len(probe.answers) == 3 and all(probe.answers)
              and f"FINAL_CODE={probe.expected}" in answer.replace(" ", "")
              and result["termination"] == "agent_completed")
    report = {"passed": passed, "request_boundary_valid": audit.valid,
              "termination": result["termination"], "agent_messages": result.get("agent_messages"),
              "previous_image_checks": probe.answers, "aihub_direct": args.aihub_direct,
              "requests": audit.records}
    (args.output / "probe-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
