# Known Issues

## GPU VRAM leak in non-headless mode across environment reloads

When running **without** `--headless` (i.e., with the GUI viewport enabled), GPU VRAM usage grows each time an environment is created and destroyed. After cycling through enough scenes, this will eventually cause an out-of-memory crash.

**Root cause:** In GUI mode, the Omniverse Kit viewport creates a Hydra render product and associated GPU textures to display the scene. When `env.close()` is called, IsaacLab clears the `SimulationContext` singleton and deletes the `ViewportCameraController`, but the **underlying Kit viewport and its Hydra render product persist** across stage reloads. Each subsequent `create_env()` call triggers `omni.usd.get_context().new_stage()`, which allocates new Hydra scene delegates and GPU-side textures for the viewport without fully releasing the previous ones. This is an IsaacLab 2.2.0 / Omniverse Kit issue — the viewport lifecycle is not tied to the `SimulationContext` lifecycle.

In headless mode, the viewport context and window are never created (`_viewport_context = None`, `_viewport_window = None`), so there is no viewport render product to leak.

**Workaround:** Always use `--headless` when running multi-task evaluations that cycle through many environments. If you need the GUI for debugging, limit your run to a small number of scenes (roughly fewer than 10, depending on GPU VRAM and camera resolution).

## Rendering artefacts

When a new environment is created and loaded, artefacts from the previous scene will remain and disappear slowly. This is an IsaacLab/RTX issue. Please refer to appropriate bug reporting for IsaacLab/Sim sources.

By using [Isaac Sim/Lab recommended GPUs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html), this effect is rarely observed.

Example run on a 3090:
<video src="https://github.com/user-attachments/assets/beb3ef86-9c57-454a-bc6c-48e75390fd12" controls width="800"></video>
