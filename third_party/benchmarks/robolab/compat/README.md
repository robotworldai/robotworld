# External Isaac Sim 6.0.1 Compatibility Layer

Visible `--isaac601` enabled, only for experimental paths. The mirror retains the official IsaacLab 2.2 source code (old WXYZ four-digit, controller and contact judge) and does not follow IsaacLab 3 from the host mirror. Source code from standard mirror, copy in `third_party/dependencies/isaaclab22`.

`isaac601.py` loads 6.0.1 with its own deprecated Core prim extensions, providing older import paths and type aliases for moving physics tensor API. Do not change upstream documents. Type aliases are used to load notes from old modules and do not mean validated software tasks; The current validation range is the DROID tool classification scenario.

No equivalent is claimed to the official Isaac Sim 5.0 baseline. The actual status and error evidence are available in parent directory `STATUS.md`.

The old TiledCamera tiled-render product returned an empty buffer on this 6.0.1 host, previously causing an out-of-bounds Warp reshape. The current single-environment test path is read by IsaacLab 2.2 with its own normal Camera, with the same camera prim, inline, position, resolution; This means that the official pixel equivalent is not supported. No false images enabled or successful trials jumped.
