"""CPU checks of task boundary and native terminal predicate delegation."""
import importlib
import json
from pathlib import Path
import types
from unittest.mock import patch

from . import project


def test_manifest_matches_native_budget_and_absolute_action():
    row = json.loads((project.ROOT / "project.json").read_text())["tasks"]["T03"]
    assert row["steps"] * row["physics_dt"] * row["decimation"] == 4.0
    assert row["id"] == project.TASK
    assert "absolute" in row["action"]


def test_success_uses_native_predicate_not_reward_or_contact():
    import numpy as np
    env = object()
    module = types.SimpleNamespace(task_completed=lambda actual: np.array([actual is env]))
    with patch.object(importlib, "import_module", return_value=module) as load:
        assert project.success(env) is True
        load.assert_called_once_with(project.PREFIX + ".mdp.terminations")


def test_prompt_discloses_prediction_and_capture_semantics():
    prompt = project.instruction("T03", None)
    for required in ("ABSOLUTE", "WXYZ", "virtual capture zone", "not a pure-vision",
                     "Future launch timer", "task_phase == 4", "100 control steps"):
        assert required in prompt
