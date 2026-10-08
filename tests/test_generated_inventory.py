import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_action_contract_inventory_matches_scoring_registry():
    scoring_tasks = load("docs/scoring/task-scoring.json")["tasks"]
    index = load("docs/action-contracts/index.json")
    validation = load("docs/action-contracts/validation.json")

    expected = {}
    for row in scoring_tasks:
        expected.setdefault(row["benchmark"], []).append(row["task"])
    actual = {row["benchmark"]: row["tasks"] for row in index["entries"]}

    assert index["tasks"] == len(scoring_tasks)
    assert sum(row["task_count"] for row in index["entries"]) == index["tasks"]
    assert actual == expected
    assert validation["selected_tasks"] == len(scoring_tasks)
