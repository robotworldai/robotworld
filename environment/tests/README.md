# Validation layers

- Contract tests: fake environments/transports for lifecycle, serialisation, timeouts, and cancellation.
- Regression tests: compare observations, targets, planner output, trajectories, and gripper timing against the reference implementation.
- Integration tests: dynamic tools and image round trips through the real source-built app-server.
- Smoke tests: short fixed-seed simulator episodes with native outcomes.

Declare GPU, asset, and credential requirements. Report unrun tests as unrun. Unit tests that use fake transports do not consume model API quota and must not modify upstream source.
