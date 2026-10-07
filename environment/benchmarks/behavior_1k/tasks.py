"""Pinned native train-instance selections, not exact Feishu instance claims."""
from pathlib import Path
import hashlib


TASKS = {
    1: dict(activity="carrying_in_groceries", scene="house_double_floor_lower",
            template_sha256="d391bcfb6cc5c5e60bca8f7eae9ccfa3cd072e743d8a7d73b1afcae392000d0c"),
    2: dict(activity="clean_up_your_desk", scene="house_single_floor",
            template_sha256="60247d1168d7b0e2ab0900f49c4e1b9171c0c7edcb096031bb212c7685f7331b"),
    3: dict(activity="slicing_vegetables", scene="house_single_floor",
            template_sha256="ab18a8f12af764e8f1b712c5f88b4e37bcc2bae5f5de5086d667d81674270061"),
    4: dict(activity="sorting_vegetables", scene="house_single_floor",
            template_sha256="43041a8f096bd6594eb2744bc8870040981ec636d7997ba22b79c85b6ef5a104"),
}


def selection(feishu_id):
    task = dict(TASKS[feishu_id])
    scene, activity = task["scene"], task["activity"]
    task.update(feishu_id=feishu_id, instance=0, split="train", seed=0, robot="R1Pro",
                gpu_dynamics=feishu_id == 3, exact_feishu_instance_verified=False,
                template=f"{scene}/json/{scene}_task_{activity}_0_0_template.json")
    return task


def verify_template(data_path, task, receipt):
    if (receipt.get("kind") != "partial_zip_members" or receipt.get("status") != "materialized"
            or task["feishu_id"] not in receipt.get("tasks", []) or not receipt.get("all_systems_included")):
        raise ValueError("Selected task lacks a validated partial asset receipt")
    path = Path(data_path) / "2026-challenge-task-instances/scenes" / task["template"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != task["template_sha256"]:
        raise ValueError("Selected task template hash mismatch")
    return path
