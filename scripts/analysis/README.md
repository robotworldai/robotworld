# Analysis of recorded evaluations

These scripts read existing run artifacts to produce reports. They do not control the simulator or change scoring.

`wheeledlab_custom_report.py` reads maps, trajectories, review images, and results for four custom driving scenarios and produces HTML. Missing results are marked incomplete. Videos are referenced through relative paths without copying large files.

Run from the repository root:

```bash
python scripts/analysis/wheeledlab_custom_report.py
```
