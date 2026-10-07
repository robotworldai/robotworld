# Isaac Sim 6.0.1 迁移

## 最新结果

用户已明确授权外部兼容代码。兼容层位于 `environment/benchmarks/robodojo/compat/`，已在 Docker 中跑通源码 Codex 的四次 move_eef 调用：抬高约 9 mm、返回误差 0.8 mm、夹爪保持打开、越界目标零执行。证据为 `World/var/runs/docker/isaac601-conveyor/attempt05/episode.json` 与 `smoke-validation.json`。完整任务不在此 smoke 中判成功。下文“尚未授权/实施”的段落保留为历史过程，由本节和 compat/README.md 的当前说明覆盖。

用户于 2026-09-25 指定使用 Isaac Sim 6.0.1；后续以此为目标，不再推进旧的 5.1 构建，不切换宿主机驱动。Python 分发版本为 `isaacsim==6.0.1.0`，不要把本机 `isaaclab==6.1.11` 误当作 Isaac Sim 6.1。

## 已验证

在 CUDA 12.8.1 base Docker 容器中只读挂载现有 Python 3.12 / Isaac Sim 6.0.1 环境和本地系统共享库，以当前 RTX 5090 / 595.80 驱动运行原生 `SimulationApp(headless=True)`。启动、一次 `app.update()`、正常关闭均成功，退出码 0。

记录：`World/var/runs/docker/isaac601-minimal/{command.json,launcher.log,engine-pass.json,exit.json}`。这验证了最小引擎运行，没有验证场景相机、物理、CuRobo 或机器人动作。该容器依赖本机只读挂载，不是可以分发的完整镜像。

## 仍须解决

追加挂载新版 IsaacLab 源码并准备检查 API 的第二次启动以 139 退出，未生成 `api-check.json`；日志仅记录 Kit 启动参数，不能据此认定具体 API 导致崩溃。最小引擎的一次成功不能替代完整组合和重复启动验证。

随后使用原始最小命令再次启动成功，退出码 0，证据为 `repeat-launcher.log`、`repeat-exit.json`、`engine-repeat-pass.json`。因此当前是最小引擎两次通过、追加 IsaacLab 挂载的组合一次启动失败；不是完整 RoboDojo 已通过。

RoboDojo 当前上游采用旧 IsaacLab API，包括 `isaaclab.sim.PhysxCfg`、`isaaclab.utils.configclass`、`SimulationCfg(physx=...)` 和 `sim.has_gui()`。此前 6.0.1 搭配 RoboDojo 原版 IsaacLab，在 `omni.physics.tensors.impl` 导入处失败。切换新版 IsaacLab 也需要验证这些调用以及姿态顺序、张量类型、机器人控制和相机。

Codex 和 benchmark 源码保持不改，不自动采用 TraceHarness 的运行时 monkey patch。尚未找到无需修改环境实现即可兼容的完整依赖组合；如必须迁移环境内部接口，应明确提出该边界冲突，不能把改写内部实现称为普通动作桥接。

## 后续验收顺序

### 追加定位：新版 IsaacLab 能启动，但旧 API 不兼容

将 API 探针从多行 `python -c` 改为脚本文件启动后，容器正常退出（0），并生成 `api-check.json`：`DirectRLEnvCfg`、`configclass` 导入通过；`from isaaclab.sim import PhysxCfg, SimulationCfg` 报 `ImportError: cannot import name 'PhysxCfg'`。这说明前一次 139 不能作为“新版 IsaacLab 一加载就崩溃”的结论；更换启动形式后的结果才是当前可复现的 API 证据。可维护探针为本目录 `isaac601_api_probe.py`。

现有新版 IsaacLab commit 为 `28a37cecdd433c22d9eabd6a5954add9f13a8951`。也读取了上游 `v3.0.0-beta2` 的 `sim/__init__.py` 和 `simulation_cfg.py`：后者同样采用 `physics: PhysicsCfg`，并非 RoboDojo 使用的 `physx=`。RoboDojo 远端当前仅公布 main 分支，HEAD `726e9aabfaa642203722eb126f5eaf0f37f3e1ad` 仍使用旧 API。没有找到可直接替换即可运行的已验证版本组合。

### 可审阅的兼容层范围（尚未实施）

若允许外部运行时兼容层，代码仅放 `environment/benchmarks/robodojo/compat/isaac601.py`，通过显式启动选项启用并记录版本：

- 恢复 RoboDojo 需要的 PhysxCfg 导入位置，将配置 `physx` 转换为新版 `physics`。
- 处理 SimContext 查询方法/属性及物理上下文访问接口变化。
- 对旧 Core prim 与新版 Lab 的物理视图类型进行适配，显式验证 numpy/Warp 的边界。
- 在机器人初始化及反馈边界转换四元数顺序，保持 RoboDojo/CuRobo 对外原顺序。
- 保持任务、资产、奖励、终止条件、动作语义和 Codex 不变；需通过实际场景/轨迹验证，不能预先宣称官方评测等价。

这涉及运行时接口替换，即 monkey patch，并不只是普通 move_eef 桥接。`environment/docs/upstream-policy.md` 明确排除了此方式，因此在明确调整该边界前不启用。TraceHarness 中已有类似代码仅作只读参考，未导入执行。

1. 确认兼容的 IsaacLab / CuRobo 版本并锁定；引擎固定 6.0.1。
2. 构建独立、可分发的 6.0.1 镜像，避免复用旧上游 Dockerfile 中固定的 5.1 依赖。
3. 加载单个 conveyor layout，获取原生观测，验证相机和物理步进。
4. 通过源码构建的 Codex app-server 调用 `move_eef`，验证位移、夹爪、非法目标拒绝。
5. 运行官方 episode，保留动作、图像与官方结果。

保留原 RoboProbe 的工具语义、坐标转换和动作执行规则；模型决策仍经过本地 Codex 源码构建产物。资产继续使用已复制的单场景子集。
