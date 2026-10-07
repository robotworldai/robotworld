# Docker 环境

镜像保存在 Docker registry / 本机 Docker 存储中，不将 docker save 的 tar 上传 GitHub。先根据 NVIDIA 官方指南配置 NVIDIA Container Toolkit，并验证 Docker 能访问 GPU。

## RoboCasa：独立构建

```bash
python3 scripts/fetch_sources.py robocasa robosuite
python3 third_party/benchmarks/robocasa/docker/build.py
```

生成 `world/robocasa:1.0.1`，不依赖 Isaac Sim；配方、依赖版本和入口见该目录 README。部分间接依赖未完全锁定，不承诺镜像逐字节复现。

## RoboLab：官方运行时与实验运行时

```bash
python3 scripts/fetch_sources.py robolab
# 官方 IsaacLab 2.2 / Isaac Sim 5.0 基础镜像
 docker build -t world/robolab:0.3.1-isaac5.0 -f third_party/benchmarks/robolab/docker/Dockerfile third_party/benchmarks/robolab/docker
```

本机 RTX 5090 / 特定驱动下官方 5.0 路径曾在启动阶段崩溃。实验镜像保留原 IsaacLab 2.2 控制层，以外部兼容层运行于 6.0.1：

```bash
docker build -t world/robolab:0.3.1-isaac6.0.1-experimental -f third_party/benchmarks/robolab/docker/Dockerfile.isaac601 third_party/benchmarks/robolab/docker
```

**前提**：上面官方镜像和 `world/robodojo:isaac6.0.1-local` 都已存在。不能把这个实验构建命令描述为新机器上完全独立的一键安装。默认 run.py 不带 --isaac601 走官方版本。

### 官方 5.1 配套环境的独立探针

另有固定 NVIDIA IsaacLab 2.3.2 镜像的 [构建与探针命令](../third_party/benchmarks/robolab/docker/README.md#官方-isaac-sim-51--isaac-lab-232-重试)。`run_probe.py --isaac51` 不加载 6.0.1 兼容层；完整入口可通过 `run.py --image world/robolab:0.3.1-isaac5.1` 指定镜像。统一 suite 的默认官方运行时仍为 5.0，未自动切换到 5.1。

2026-09-27 实测：RTX 5090 / 驱动 595.80 上，5.1 镜像构建成功，但纯 SimulationApp 和带相机的 AppLauncher 均在 `rtx.scenedb` 初始化崩溃；尚未进入场景或物理步进。因此未将 5.1 标为运行通过。Docker 共用宿主机 NVIDIA 内核驱动，不能靠在单个容器内安装旧驱动独立切换。见 [STATUS.md](../third_party/benchmarks/robolab/STATUS.md)。

## BEHAVIOR-1K

```bash
python3 scripts/fetch_sources.py behavior_1k
docker pull stanfordvl/behavior:3.9.3
```

实验 6.0.1 配方位于 `environment/containers/behavior_1k/Dockerfile.isaac601`，依赖 RoboDojo 本地快照基础镜像。运行和差异见该目录 README；官方资产单独准备。

## RoboDojo

源码在 `third_party/benchmarks/RoboDojo/`；运行、旧 5.1 配方、6.0.1 快照构建和外部兼容说明在 `environment/containers/robodojo/`。

**当前交付限制**：已验证的 `world/robodojo:isaac6.0.1-local` 来自作者主机已安装环境，`build_isaac601_snapshot.py` 与 SNAPSHOT_BUILD.md 是快照提取配方，可能引用作者本机路径；这些依赖未随代码提供。新机器须自行准备匹配运行时，或由维护者另行发布获准分发的基础镜像。本代码包没有宣称这一镜像已公开。由其派生的 BEHAVIOR / RoboLab 实验镜像同样受此限制。

## HumanoidSoccer

`bash scripts/eval/humanoid_soccer.sh build` 构建独立 `world/humanoid-soccer:mujoco3.3.1`，不依赖 Isaac Sim/其他 bench 镜像。需要 NVIDIA Container Toolkit 提供 EGL。完整命令见 [Docker 说明](../third_party/benchmarks/humanoid_soccer/docker/README.md)。


## 其余项目与本轮使用的配置

下面逐项从固定 project.json / runtime-profiles 解析，列的是本轮诊断选择，不表示全部启动成功。各自源码、资源准备和限制在 third_party/benchmarks/项目名/README.md。T02/T14/T16/T17和单机颠球必须显式选择isaac6，不能把默认旧版镜像和实验镜像混算。

| 项目 | runtime-profile | 镜像 | Dockerfile |
|---|---|---|---|
| bench2dex | anchored | `world/bench2dex:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/bench2dex/docker/Dockerfile) |
| steadytray | default | `world/steadytray:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/steadytray/docker/Dockerfile) |
| ttrl | isaac6 | `world/ttrl:isaac6.0.1-experimental` | [Dockerfile.isaac6](../third_party/benchmarks/ttrl/docker/Dockerfile.isaac6) |
| reflexbench | default | `world/reflexbench:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/reflexbench/docker/Dockerfile) |
| aerial_balance | default | `world/aerial-balance:isaac4.2.0` | [Dockerfile](../third_party/benchmarks/aerial_balance/docker/Dockerfile) |
| volleybots | default | `world/volleybots:isaac2023.1.0-hotfix1` | [Dockerfile](../third_party/benchmarks/volleybots/docker/Dockerfile) |
| volleybots | isaac6 | `world/volleybots:isaac6.0.1-experimental` | [Dockerfile.isaac6](../third_party/benchmarks/volleybots/docker/Dockerfile.isaac6) |
| wheel_legged | default | `world/wheel-legged:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/wheel_legged/docker/Dockerfile) |
| wheeled_quadruped | default | `world/wheeled-quadruped:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/wheeled_quadruped/docker/Dockerfile) |
| go2_push | default | `world/go2-push:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/go2_push/docker/Dockerfile) |
| robot_lab | a1-feet | `world/robot-lab:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/robot_lab/docker/Dockerfile) |
| digit | default | `world/digit:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/digit/docker/Dockerfile) |
| omniisaacgymenvs | isaac6 | `world/omniisaacgymenvs:isaac6.0.1-experimental` | [Dockerfile.isaac6](../third_party/benchmarks/omniisaacgymenvs/docker/Dockerfile.isaac6) |
| flamingo | default | `world/flamingo:isaac6.0.1-experimental` | [Dockerfile](../third_party/benchmarks/flamingo/docker/Dockerfile) |
| omnidrones | isaac6 | `world/omnidrones:isaac6.0.1-experimental` | [Dockerfile.isaac6](../third_party/benchmarks/omnidrones/docker/Dockerfile.isaac6) |

AI-CPS 使用 `world/ai-cps:isaac6.0.1-experimental`，配方为 [AI-CPS Dockerfile](../third_party/benchmarks/ai_cps/docker/Dockerfile)。WheeledLab使用 `world/wheeledlab:isaac6.0.1-experimental`，配方为 [WheeledLab Dockerfile](../third_party/benchmarks/wheeledlab/docker/Dockerfile)。都通过各自 `scripts/eval/项目名.sh build` 构建，所需基础镜像须先具备。

例如复现实验版乒乓环境：

```bash
bash scripts/eval/ttrl.sh build --runtime-profile isaac6
bash scripts/eval/ttrl.sh probe --runtime-profile isaac6 --output var/runs/ttrl-check
```

T04的原资产缺失、T05原1v1缺固定完整对手、T15原传感器配置错误不会通过构建镜像自动消失；见逐任务核验报告。本轮镜像ID记录在 reports/validation/2026-09-29/provenance-after-migration.json，tag不是物理等价承诺。
