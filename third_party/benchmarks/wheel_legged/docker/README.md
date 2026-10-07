# Containers

From World root directory:

```bash
docker build -t world/wheel-legged:isaac6.0.1-experimental \
  -f third_party/benchmarks/wheel_legged/docker/Dockerfile \
  third_party/benchmarks/wheel_legged/docker
```

Reuse existing world/wheeledlab mirrors and IsaacLab2.2 overlay; Isaac is not duplicated. The source code is placed in the container by a read-only mounted, and the robot URDF converts USD to run/generated_assets. Codex runs local app-server compiled by World/codex in host isolation sandbox, and only socket is called by the container; Not copy API key in the container, not call the model HTTP API directly.

The URDF conversion retains upstream fix_base, collision and kinetic settings. When the Isaac6 interface is not compatible, only external API is allowed to fit and no alternative robots/scene/noise-free Play is allowed to mask failure. The currently used external compatibility position is third_party/benchmarks/robolab/compat/isaac601.py.

Source of actual dependence: Original code only imports basic components such as IsaacLab, torch; Evaluate not to import RSL-RL trainers or checkpoint. Third parties rely on the provision of basic mirrors. Upstream source code and licence remain based on checkout content.
