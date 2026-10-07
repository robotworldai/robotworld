# Isaac Sim 5.1 RTX 启动诊断（2026-09-25）

## 实验结论

故障已隔离到 Isaac Sim 的 RTX 初始化路径：不导入 RoboDojo、不启动 Codex、不加载场景，只有 `SimulationApp({"headless": True})` 的最小程序，也在 `librtx.scenedb.plugin.so` 的同一调用栈崩溃。

宿主是 RTX 5090 / NVIDIA Open Kernel Module 595.80 / Ubuntu 22.04 / Linux 6.8.0-40。容器的 CUDA 和 Vulkan 都识别到 GPU。补齐 libxml2 及递归依赖、将 Kit data/cache/logs 放入容器 tmpfs 后，原来的缺库、只读目录错误消失，崩溃仍复现。日志中另有 requests 依赖警告，但不是原来的动态库加载失败。

证据（相对 World）：

- `var/runs/docker/conveyor-local-env/launcher-clean.log`：补齐依赖后的 RoboDojo 入口，仍在 app ready 后崩溃。
- `var/runs/docker/conveyor-local-env/minimal-isaac.log`：不加载 benchmark 的最小对照实验。
- `var/runs/docker/conveyor-local-env/minimal-isaac-command.json`：完整 argv；只读挂载原版 Python 环境，没有上游代码补丁。
- `var/isaac5-system-libs/manifest.json`：41 个本地系统库来源和 SHA256。

## 与上游问题的对应

[IsaacSim #687](https://github.com/isaac-sim/IsaacSim/issues/687) 报告 RTX 5090、595.71.05、5.1.0 的同类崩溃；[IsaacSim #619](https://github.com/isaac-sim/IsaacSim/issues/619) 报告原生与官方 Docker 都复现，堆栈包含相同的 `carbOnPluginStartup+0x3b4de`。

[Isaac Sim 5.1 官方要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) 列出的 Linux 测试驱动是 580.65.06。[NVIDIA 论坛](https://forums.developer.nvidia.com/t/isaacsim-crash-when-update-gpu-driver/371975) 也记录了较新驱动分支与 5.1 RTX 启动的不兼容。

因此当前首要假设是 **595.80 与 Isaac Sim 5.1 RTX 的兼容性**，证据强，但未做驱动切换 A/B，不能声称已经证明更换某个版本必定修复。继续下载 CUDA devel 镜像本身不会更换宿主内核驱动。

## 可执行的下一步及边界

优先在使用合适 R580 open 驱动的空闲机器验证最小程序，再运行同一 conveyor 资产包和 Codex 工具入口。这样不需要改 RoboDojo/IsaacLab/Codex 源码。

若必须使用当前主机，需要管理员安排驱动维护窗口：

1. 与当前 GPU 任务的所有者协调停止任务并保存状态；当前有两个其他计算进程，未对它们操作。
2. 备份驱动安装记录、准备可回退的 595.80 安装介质，并确认本地/远程恢复控制台。
3. 确认现有驱动安装方式。系统有 `/usr/bin/nvidia-uninstall`，dpkg 未显示 NVIDIA 驱动元包；可能是 .run 安装，不能直接叠加 apt 驱动而忽略残留文件。
4. 选择支持 RTX 5090 的 R580 **open kernel module** 驱动，并审核其余 CUDA 工作负载兼容性。APT 当前候选为 `nvidia-driver-580-open=580.178.04-0ubuntu0.22.04.1`，它不是官方表中测试的同一 patch 版本，仍须实测。
5. 完成安装和重启后，依次验证 nvidia-smi、Docker GPU、最小 Isaac 渲染、单场景 probe、源码 Codex 官方任务。

已执行的只是 `apt-get -s install nvidia-driver-580-open` 模拟，输出在 `var/runs/docker/conveyor-local-env/driver-580-dry-run.txt`。它计划升级 16 包、安装 63 包，未实际改动系统。没有自动卸载驱动、停止其他进程或重启。

Isaac Sim 6 在当前驱动可启动，但原 RoboDojo IsaacLab 使用的 `omni.physics.tensors.impl` 已不存在；采用已知 TraceHarness 兼容补丁会违反“环境源码和运行时不打补丁”的约束。不能将直接切到 6 当作已经可用的替代方案。
