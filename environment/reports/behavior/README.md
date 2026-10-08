# BEHAVIOR review page

Static, allowlisted review of the first-four diagnostic. The visual treatment
follows the supplied VideoCut reference: dark-blue cards, video results,
tool/observation outline, expandable parameters and raw display records.

Export with the host report/Scaffold venv, never the simulation environment:

```bash
python environment/reports/behavior/export.py \
  --batch /path/to/local/batch \
  --output /path/to/report-only/robotworld-behavior
python environment/reports/behavior/serve.py --root /path/to/report-only --port 8080
```

The served root contains ONLY report files. No symlinks to private runs. The
server refuses directory listings, symlinks, traversal, arbitrary extensions
and writes, and supports MP4 byte ranges. The route has no authentication;
publish only to the intended internal network. Model explanations are redacted,
and API payloads, credential files, source paths and asset archives are omitted.

The exporter correlates native actions using input observation IDs, verifies
arguments and cumulative native step counts, and links post-action RGB. Frame
zero is after control step 1, so a post-action seek uses `(end_step - 1) / 30`.
Pure observe calls do not add physical time. No hidden reasoning or missing
tool output is fabricated. This is a static snapshot, not a live job controller.
Download JSON is the display schema, not a raw Codex session dump.

Self-host `fonts/NotoSansSC.ttf` for the reference page's CJK typeface; the CSS
falls back if unavailable. Browser dependencies live in an isolated report
venv. `browser_check.py` checks desktop/mobile layout, four videos, image loads,
seeking, camera selection and failure filtering. Unit tests live in
`environment/tests/test_behavior_report.py`.

## Offline single file

```bash
python environment/reports/behavior/bundle.py \
  --source /path/to/report-only/robotworld-behavior \
  --output /path/to/offline/robotworld-behavior.html
```

This embeds all selected videos, RGB, display JSON and CJK font. It uses local
Blob URLs for media and embedded JSON instead of fetch; a CSP blocks network
connections. Open the HTML directly in a desktop browser. Base64 increases
file size; no video is recompressed. `browser_check.py --offline` verifies the
file with browser networking disabled. Do not publish the private source root.
