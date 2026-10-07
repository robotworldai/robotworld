# Docker

`Dockerfile` returns `world/robodojo:isaac6.0.1-local`, adding only gym and independent RTAMT interpreter. `run.py` verify source code version and asset Hash, mount environment read-only, output and cache independently written, via isolated source code Codex bridging tool.

Main entrance: `bash scripts/eval/ai_cps.sh build`, `bash scripts/eval/ai_cps.sh run --model MODEL --codex-home PATH`. Run mode codex for model assessment; zero, probe, recovery-check are clearly marked infrastructure diagnostics. The last one only allows case34. Builds mirrors that do not contain Codex authentication, assets or turn logs.
