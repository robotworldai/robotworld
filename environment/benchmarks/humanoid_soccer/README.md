# G1 control adapter

[CODING_CONTROL.md](CODING_CONTROL.md) describes generated control. `coding_control` executes the restricted AST language in `control_program.py`; `coding.py` and `program_worker.py` manage a separate worker and resource limits. The schema is in `control.py::coding_spec`, and the system instructions are in `policy.py::CODING_INSTRUCTIONS`.

`scenery.py` adds an optional rendering-only training-field background. `balance.py` provides explicitly enabled experimental ankle/centre-of-mass assistance for direct control. Neither edits upstream source or scene physics, but assistance changes actions and must be reported as a custom-controller diagnostic.

`control.py` defines mode-specific tools, validation, and action ordering. `policy.py` handles the local source-built agent session, upstream proposal review, direct joint control, observation history, and event records. Control state is not exposed through agent file commands.

Upstream code, image recipes, assets, and the [run guide](../../../third_party/benchmarks/humanoid_soccer/README.md) are in `third_party/benchmarks/humanoid_soccer/`. Evaluation retains the upstream `run_trial` and scoring methods.
