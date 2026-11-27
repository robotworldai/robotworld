"""Portable entry point; generated tasks bundle the canonical shared grader."""
import importlib.util
from pathlib import Path

_path = Path(__file__).with_name('shared_grade.py')
_spec = importlib.util.spec_from_file_location('robotworld_shared_grader', _path)
_grader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grader)

# Compatibility for diagnostics and existing result-inspection tools.
InvalidTrial = _grader.InvalidTrial
validate = _grader.validate
audit_artifacts = _grader.audit_artifacts
read_jsonl = _grader.read_jsonl
verify_video = _grader.verify_video
main = _grader.main

if __name__ == '__main__':
    main()
