# Native control timing audit

Task configuration requests `cfg.sim.dt=0.016`, `substeps=1`. The original
Isaac Sim4.1 image's `/isaac-sim/exts/omni.isaac.core/omni/isaac/core/physics_context/physics_context.py`
does `steps_per_second = int(1.0 / dt)` and writes that integer to
`PhysxSceneAPI.timeStepsPerSecond`. Its getter returns `1.0 / physics_hz`.
Thus the task's own `self.dt = self.sim.get_physics_dt()` is **1/62 seconds**,
not the nominal YAML value. Isaac6 has the same integer conversion.

The World adapter now reports the actual original `env.dt` and records
`configured_physics_dt` separately. It does not change the requested YAML
or task dynamics equations. Push probabilities, trajectories and native
rewards continue to use the original task's own timestep.

In the experimental Isaac6 profile, probe10 demonstrated that the render=True
Core path (`app.update`) could produce zero physics ticks after annotator
initialization. Each original `sim.step` is therefore bridged to one explicit
Core physics-only tick plus a render-only refresh. Per-tick and per-action
engine time deltas are asserted; rendering must not increment physical time.
This is an external runtime compatibility path, not an additional controller.

500 steps = 8.064516129 seconds; 600 steps = 9.677419355 seconds.
Review video uses 62 FPS and contains the initial frame plus one frame per
executed action step. Paused model reasoning never generates physical steps.
