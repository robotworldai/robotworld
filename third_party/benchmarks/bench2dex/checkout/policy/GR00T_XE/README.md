# GR00T XE (Cross-Embodiment)

GR00T N1.5 is fine-tuned by cross-sweet hand, based on [NVIDIA Isaac GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T).

## Overview

GR00T XE unifies teleoperation data from 12 dexterous hands and 26 tasks into a 64-dimensional action space:

| Part | Dimensions | Annotations |
|------|------|------|
| Right hand arm ee pose | 6 | End position + Opposite Ori. |
| Left hand arm ee pose | 6 | Ibid. |
| Right hand hand | 22 | Semantic slots ( thumb/finger/mode/unname/finger/hand) |
| Left hand hand | 22 | Mirror |
| padding | 8 | 64 |

## Training

### Phase 1: Pretrain (all jobs)

```bash
bash policy/GR00T_XE/pretrain.sh
```

### Phase 2: Per-task Finetune

```bash
bash policy/GR00T_XE/finetune.sh <TASK_ID> --tag <TAG>
```

## Evaluation

```bash
bash policy/GR00T_XE/eval_double_env.sh <TASK_ID> [none|cov_only|inv_only|inv_cov]
```

## Environment

Shares `groot` conda environment with GR00T_n15. For first use to run:

```bash
bash policy/GR00T_n15/setup_env.sh
```

## File Structure

```
policy/GR00T_XE/
├── pretrain.py / pretrain.sh      # Full Task pretrain
├── finetune.py / finetune.sh      # Single task finetune
├── deploy_policy.py               # Deployment: Load Model + FK/IK Convert
├── deploy_policy.yml              # Deployment Configuration
├── ik_arm_converter.py            # FK/IK + Hand map converter
├── embodiment_mapping.yml         # 12 Hands. → 44 Semantic Slot Map
├── xe_config.py                   # Data Configuration
├── dataset.py                     # Dataset
├── gr00t_hdf5_dataset.py          # HDF5 Data Loading
├── eval_double_env.sh             # Evaluate Script
├── src/                           # GR00T Model Extension
├── LICENSE                        # Apache 2.0
├── requirements.txt               # Dependency
└── setup_env.sh                   # Environmental installation
```

## Licence

Apache 2.0, matching upstream NVIDIA Isaac GR00T.

## References

```bibtex
@inproceedings{gr00tn1_2025,
  title  = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author = {NVIDIA et al.},
  year   = {2025},
  booktitle = {ArXiv Preprint},
}
```