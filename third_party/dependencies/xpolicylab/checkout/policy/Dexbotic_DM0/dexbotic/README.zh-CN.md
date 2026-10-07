<div align="center">
  <img src="resources/logo.png" alt="Dexbotic Logo" width="280"/>

  # One-stop smart VLA development toolbox

  [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.23511)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/Dexmal)
  [![Documentation](https://img.shields.io/badge/Docs-Online-success)](https://dexbotic.com/docs/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![English](https://img.shields.io/badge/lang-English-red.svg)](README.md)

  <p align="center">
    <strong> Pre-training — fine-tuning — reasoning — </strong> <br>
    Supports mainstream policies such as 0, CogACT, OFT, MemVLA
  </p>
</div>

## Introduction

** Dexbotic** is a VLA (Visual-Language-Action) development toolbox based on the PyTorch framework designed to provide a unified and efficient solution for smart research. It embeds the environmental configuration of many mainstream VLA models, and users can repeat, fine-tune and deduce a variety of frontier algorithms with simple settings.

- ** opens the VLA frame **: with the VLA model at its core, it combines hands-on and navigation functions to support leading algorithms in multiple industries.
- **Pretrained foundation models:** provides Dexbotic-optimised pretrained models for VLA architectures including π0 and CogACT.
- ** Modularized Development Architecture **: Using the "Stage Configuration + Plant Registration + Entry Distribution" structure, users simply need to modify their experimental scripts so that they can easily meet needs such as configuration modifications, model changes or task additions.
- ** Cloud and Local Integration Training **: Comprehensive support for cloud and local training needs, cloud training platforms such as Aliyun, Volcanic Engines, and local training adapted to consumer level GPU.
- **Robot support:** provides a **shared training-data format** and deployment scripts for robots including UR5, Franka, and ALOHA.

![](resources/intro.png)

## Recent developments

- ** [2026-03-30] ** supports [GR00TN1](playground/benchmarks/libero/libero_gr00tn1.py) models.
- ** [2026-03-30] ** New Pi05 Model 's [Joint training](dexbotic/exp/hybrid_pi05_exp.py) capability.
- ** [2026-03-30] ** has published the [XLeRobot](hardware/docs/xlerobot_inference_example.md) tutorial used in Dexbotic.
- ** [2026-02-10] ** [DM0](docs/DM0.md) Published! See [Technical report](https://dexmal.com/DM0_Tech_Report.pdf) for details.
- ** [2026-02-10] ** Cooperation Bulletin: We are very pleased to announce strategic cooperation with [RLinf](https://github.com/RLinf/RLinf). VLA + RL will jointly advance research and applications.
- ** [2026-01-15] ** has published the [SO-101](hardware/docs/so101_inference_example.md) tutorial used in Dexbotic.
- ** [2026-01-15] ** supports [GRPO](docs/RL.md).
- ** [2026-01-15] ** supports [NaVILA](playground/example_navila_exp.py).
- ** [2026-01-08] ** adds a [Joint training](dexbotic/exp/hybrid_cogact_exp.py) capability to support joint optimization of CogACT model action experts with LLM.
- ** [2026-01-08] ** has released a unique mirror suitable for [Blackwell GPU](#blackwell-gpus).
- ** [2025-12-29] ** supports [OFT](playground/benchmarks/libero/libero_oft.py) and [Pi0.5](playground/benchmarks/libero/libero_pi05.py) models.
- ** [2025-10-20] ** Dexbotic Published! See [Technical report](https://arxiv.org/pdf/2510.23511) and [Official documents](https://dexbotic.com/docs/) for further information.


## Fast start.

We strongly recommend the use of Docker for development or deployment to obtain the best use experience.

### 1. Installation and environmental configuration

```bash
# 1. Cloning code repository
git clone https://github.com/dexmal/dexbotic.git

# 2. Start Docker Containers
docker run -it --rm --gpus all --network host \
  -v $(pwd)/dexbotic:/dexbotic \
  dexmal/dexbotic \
  bash

# 3. Activate environment and install dependency
cd /dexbotic
conda activate dexbotic
pip install -e .
```
> The ** system requires **: Ubuntu 20.04/22.04 and recommends RTX 4090, A100 or H100 (training recommendation 8 GPU for deployment with 1 GPU).

<details id="blackwell-gpus">
<summary>Using Blackwell GPUs</summary>

For Blackwell GPUs such as B100 and RTX 5090, use the dedicated image `dexmal/dexbotic:c130t28`.

```bash
# 1. Use Blackwell Mirror activated. Docker
docker run -it --rm --gpus all --network host \
  -v /path/to/dexbotic:/dexbotic \
  dexmal/dexbotic:c130t28 \
  bash

# 2. Activate Environment**
cd /dexbotic
pip install -e .
```

</details>

### 2. User Guide

- [Testing and evaluation](docs/Tutorial.md#evaluation)
- [Training based on simulation data](docs/Tutorial.md#training-a-model-with-provided-data)
- [Training in the use of own data](docs/Tutorial.md#training-a-model-with-your-own-data)


## Benchmark testing

The following is a comparison of the model based on Dexbotic training with the results of the evaluation of the original model in the mainstream simulation environment. ** to see more detailed evaluation results **: [Benchmark Results](docs/ModelZoo.md#benchmark-results)

### Libero

| Model | Average | Libero-Spatial | Libero-Object | Libero-Goal | Libero-10 |
| --- | --- | --- | --- | --- | --- |
| CogACT | 93.6 | 97.2 | 98.0 | 90.2 | 88.8 |
| DB-CogACT | 94.9 | 93.8 | 97.8 | 96.2 | 91.8 |
| π0 | 94.2 | 96.8 | 98.8 | 95.8 | 85.2 |
| DB-π0 | 93.9 | 97 | 98.2 | 94 | 86.4 |
| MemVLA | 96.7 | 98.4 | 98.4 | 96.4 | 93.4 |
| DB-MemVLA | 97.0 | 97.2 | 99.2 | 98.4 | 93.2 |
| DB-GR00TN1 | 94.8 | 93.0 | 99.6 | 95.2 | 91.4 |

### CALVIN

| Model | Average Length | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| CogACT | 3.246 | 83.8 | 72.9 | 64 | 55.9 | 48 |
| DB-CogACT | 4.063 | 93.5 | 86.7 | 80.3 | 76 | 69.8 |
| OFT | 3.472 | 89.1 | 79.4 | 67.4 | 59.8 | 51.5 |
| DB-OFT | 3.540 | 92.8 | 80.7 | 69.2 | 60.2 | 51.1 |

### SimplerEnv

| Model | Average | Spoon | Carrot | Stack Blocks | Eggplant |
| --- | --- | --- | --- | --- | --- |
| CogACT | 51.25 | 71.7 | 50.8 | 15 | 67.5 |
| DB-CogACT | 69.45 | 87.5 | 65.28 | 29.17 | 95.83 |
| OFT | 30.23 | 12.5 | 4.2 | 4.2 | 100 |
| DB-OFT | 76.39 | 91.67 | 76.39 | 43.06 | 94.44 |
| MemVLA | 71.9 | 75.0 | 75.0 | 37.5 | 100.0 |
| DB-MemVLA | 84.4 | 100.0 | 66.7 | 70.8 | 100.0 |

### ManiSkill2

| Model | Average | PickCube | StackCube | PickSingleYCB | PickSingleEGAD | PickClutterYCB |
| --- | --- | --- | --- | --- | --- | --- |
| CogACT | 40 | 55 | 70 | 30 | 25 | 20 |
| DB-CogACT | 58 | 90 | 65 | 65 | 40 | 30 |
| OFT | 21 | 40 | 45 | 5 | 5 | 0 |
| DB-OFT | 63 | 90 | 75 | 55 | 65 | 30 |
| π0 | 66 | 95 | 85 | 55 | 85 | 10 |
| DB-π0 | 65 | 95 | 85 | 65 | 50 | 30 |

### RoboTwin2.0

| Model | Average | Adjust Bottle | Grab Roller | Place Empty Cup | Place Phone Stand |
| --- | --- | --- | --- | --- | --- |
| CogACT | 43.8 | 87 | 72 | 11 | 5 |
| DB-CogACT | 58.5 | 99 | 89 | 28 | 18 |

## Common problems

<details close>
<summary> Q: Flash-Attention installation failed </summary>

A: For detailed installation and troubleshooting, see https://github.com/Dao-AILab/flash-attention.
</details>

<details close>
How to convert <summary> Q: RLDS/LeRobot data format to Dexdata?</summary>

A:, we have a general data conversion method in [Data conversion guide](docs/Data.md#2-data-conversion). For examples of LeRobot data conversion, see [convert_lerobot_to_dexdata](script/convert_data/convert_lerobot_to_dexdata.py), and for RLDS data conversion, see [convert_rlds_to_dexdata](script/convert_data/convert_rlds_to_dexdata.py).
</details>

<details close>
Do you support <summary> Q: 5090 graphic cards?</summary>

A: support, see [Blackwell frame graphic card use](#blackwell-gpus).
</details>

## Support us.

We are constantly improving and more functionality is about to be rolled out. If this project is useful, consider starring it on [GitHub](https://github.com/dexmal/dexbotic).

If Dexbotic helps your research, please consider quoting our technical report:

```bibtex
@article{dexbotic,
  title={Dexbotic: Open-Source Vision-Language-Action Toolbox},
  author={Dexbotic Contributors},
  journal={arXiv preprint arXiv:2510.23511},
  year={2025}
}
```

## Allow

This project uses [MIT license](LICENSE).