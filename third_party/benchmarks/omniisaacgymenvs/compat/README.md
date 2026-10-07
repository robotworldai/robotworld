# Isaac Sim 6.0.1 External Compatibility Layer

`vendor/omni.isaac.gym` extracts the full gym extension from the official Isaac Sim 4.0 mirror image installed and does not belong to benchmark checkout. Sim6 has removed this interface; Reuse old VecEnv to keep original step and RLTaskInterface. Document by document and original mirror digest see `vendor-provenance.json`.

** is used locally only and does not include open source snapshot **: the original NVIDIA licence is retained; Users should take it from their official permission mirrors and not mark it as our open source code.

The actual fit-out is at `environment/benchmarks/omniisaacgymenvs/project_isaac6.py`. Original default profile remains unchanged.
