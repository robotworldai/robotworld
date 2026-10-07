"""Utility to tag soccer kicking motions with the striking foot.

This helper reads existing ``.npz`` motion files, adds a metadata entry
(``kick_leg``), then writes a new file with a ``_left`` / ``_right`` suffix.
The original file remains unchanged. The current training pipeline (for example,
``MultiMotionLoader``) only reads state arrays, so this extra key is ignored.

Usage examples::

	python kick_motion_label.py motions/football-kjp --label left
	python kick_motion_label.py motion_a.npz motion_b.npz --label right --overwrite

The script accepts individual files or directories. When a directory is
provided, every ``.npz`` file in it is processed (use ``--recursive`` to
recurse into subfolders).
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile
from typing import Iterable

import numpy as np


LABEL_KEY = "kick_leg"


def collect_npz_files(targets: Iterable[str], recursive: bool) -> list[str]:
	"""Collect all NPZ files from the provided targets."""

	collected: list[str] = []
	for raw_target in targets:
		target = os.path.abspath(raw_target)
		if os.path.isfile(target):
			if not target.lower().endswith(".npz"):
				raise ValueError(f"Not an NPZ file: {raw_target}")
			collected.append(target)
			continue

		if os.path.isdir(target):
			pattern = "**/*.npz" if recursive else "*.npz"
			glob_pattern = os.path.join(target, pattern)
			for path in glob.glob(glob_pattern, recursive=recursive):
				if os.path.isfile(path):
					collected.append(os.path.abspath(path))
			continue

		raise FileNotFoundError(f"Path does not exist: {raw_target}")

	# Remove duplicates while preserving deterministic order.
	return sorted(set(collected))


def load_npz_payload(path: str) -> tuple[dict[str, np.ndarray], str | None]:
	"""Load an NPZ file into a plain dict and return the prior label."""

	with np.load(path, allow_pickle=False) as npz_file:
		payload = {key: npz_file[key] for key in npz_file.files}
	prior = payload.get(LABEL_KEY)
	if prior is None:
		return payload, None

	try:
		prior_label = str(prior.item())  # works for 0-d arrays
	except ValueError:
		prior_label = str(prior)
	return payload, prior_label


def write_npz_payload(path: str, payload: dict[str, np.ndarray]) -> None:
	"""Persist the payload to *path* atomically."""

	directory = os.path.dirname(path) or "."
	with tempfile.NamedTemporaryFile(delete=False, dir=directory, suffix=".tmp.npz") as tmp_file:
		tmp_path = tmp_file.name

	try:
		np.savez(tmp_path, **payload)
		os.replace(tmp_path, path)
	finally:
		if os.path.exists(tmp_path):
			os.remove(tmp_path)


def build_output_path(path: str, label: str) -> str:
	base, ext = os.path.splitext(path)
	for suffix in ("_left", "_right"):
		if base.endswith(suffix):
			base = base[: -len(suffix)]
			break
	return f"{base}_{label}{ext}"


def update_label(path: str, label: str, *, dry_run: bool, overwrite: bool) -> tuple[str, str]:
	"""Write the labeled copy for *path*. Returns status and output path."""

	payload, prior_label = load_npz_payload(path)
	output_path = build_output_path(path, label)

	existed_before = os.path.exists(output_path)
	if existed_before and not overwrite:
		return "skip (output exists)", output_path

	if prior_label is not None and prior_label != label and not overwrite:
		return f"skip (existing label: {prior_label})", output_path

	payload[LABEL_KEY] = np.array(label)

	if dry_run:
		return "dry-run", output_path

	write_npz_payload(output_path, payload)
	status = "overwritten" if existed_before else "written"
	return status, output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Tag kicking motions with left/right labels.")
	parser.add_argument(
		"targets",
		nargs="+",
		help="NPZ files or directories containing NPZ motion files.",
	)
	parser.add_argument(
		"--label",
		choices=("left", "right"),
		required=True,
		help="Which leg performs the kick in the selected motions.",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Recurse into subdirectories when a directory target is provided.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only report the changes that would be applied without touching any files.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow overwriting existing labeled copies or conflicting labels.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	try:
		files = collect_npz_files(args.targets, recursive=args.recursive)
	except (ValueError, FileNotFoundError) as exc:
		raise SystemExit(f"[ERROR] {exc}") from exc

	if not files:
		raise SystemExit("[INFO] No NPZ files found. Nothing to do.")

	print(f"[INFO] Processing {len(files)} motion file(s) with label '{args.label}'.")
	if args.dry_run:
		print("[INFO] Running in dry-run mode; no files will be modified.")

	for path in files:
		status, output_path = update_label(path, args.label, dry_run=args.dry_run, overwrite=args.overwrite)
		print(f"  - {path} -> {output_path}: {status}")


if __name__ == "__main__":
	main()
