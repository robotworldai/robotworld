# OmniIsaacGymEnvs Dependence

`checkout/` contains the pinned official `release/2022.2.0` source, with its commit in `source.json`, supplying RLTask, SimConfig, and USD helpers for AI-CPS. Keep upstream, licence in checkout. Do not place World policy, output, model authentication or environmental assets here.

Restore: `python scripts/fetch_sources.py omniisaacgymenvs`. Isaac6 is compatible with `third_party/benchmarks/ai_cps/compat/`. Only three tasks are imported to avoid non-material/training dependency.
