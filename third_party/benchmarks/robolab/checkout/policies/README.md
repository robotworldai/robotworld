# Inference Clients and Policy Server Setup

RoboLab uses a **server-client architecture**: your model runs as a standalone server process, and RoboLab connects to it through a lightweight inference client during evaluation.

For writing your own inference client, see [Evaluating a New Policy](../docs/policy.md). For the full run CLI reference, see [Running Environments](../docs/environment_run.md#run-cli-reference).

## Shipped policy clients

| Backend | Policy / model | References |
|---------|----------------|------------|
| [`pi0_family/`](pi0_family/README.md) | π0, π0-FAST, π0.5, PaliGemma, PaliGemma-FAST (select with `--policy`) | [Code](https://github.com/Physical-Intelligence/openpi), Papers: [π0](https://arxiv.org/abs/2410.24164), [FAST](https://arxiv.org/abs/2501.09747), [π0.5](https://arxiv.org/abs/2504.16054) |
| [`cosmos3/`](cosmos3/README.md) | Cosmos3-Nano-Policy | [Website](https://huggingface.co/collections/nvidia/cosmos3), [Paper](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) |
| [`gr00t/`](gr00t/README.md) | GR00T N1.7 DROID / GR00T N1.6 DROID | [Website](https://developer.nvidia.com/isaac/gr00t), [Code](https://github.com/NVIDIA/Isaac-GR00T), [Paper](https://arxiv.org/abs/2503.14734) |
| [`dreamzero/`](dreamzero/README.md) | DreamZero-DROID | [Code](https://github.com/dreamzero0/dreamzero), [Paper](https://arxiv.org/abs/2602.15922) |
| [`volo/`](volo/README.md) | VoLoAgent | [Website](https://chicychen.github.io/VoLo/), [Code](https://github.com/NVlabs/RoboVoLo), [Paper](https://arxiv.org/abs/2606.07723) |
