# 验证状态（2026-09-27）

## 后续失败案例与 1000 步诊断

- 后续 `coding_control` 实测：`seed2-coding-balance-1000steps-01`，本地源码 Codex + GPT-6 编写并调用 13 段代码；1001 次代码回调对应 1000 个动作步（一次 done 不消耗步数）。一段程序因骨盆高度下降主动提前返回；最后一段被剩余 episode 预算截断。机器人第 176 步骨盆低于 0.35m，未进球；接口跑通不代表控制器已实现稳定行走。原始代码见各回合 `agent/programs/`，程序事件与无图日志均已记录。该回合在添加 callback_wall_ms/observation 扩展日志字段前运行，因此这两个字段仅适用于后续新回合。
- Coding 工具独立测试 15 passed；仿真集成测试提前 done、非法动作、剩余预算三段实际执行 2/0/4 步，共 6 步/0.12s。代码 worker 为 AST 子集解释器，不使用 eval/exec，文件/网络/任意导入不可用，循环和资源超限作为工具错误返回。

- 原版默认分布独立测试 seed=0..19，19 次成功，seed=2 失败（碰球但未进门）。这是案例筛选集，不是 GPT 增益统计。
- seed=2 的简洁/训练球场背景视频对照，qpos/qvel/torque/time 全部逐元素相同。背景仅可视化，无碰撞或任务修改。
- seed=2 + 球场背景 + GPT hybrid：300 步失败，8 次 review_action 全部 accept，没有体现修复增益。
- 新增实验性 `--balance-assist ankle-com`（仅 direct），不改上游环境，仅改变送入原 PD 的踝关节目标。固定初始姿态保持试验：seed=2/7 无辅助均在约 1.246s 骨盆低于 0.35m；辅助均完成 1000 控制步，最低骨盆高度约 0.783m。这两个 seed 的初始机器人姿态相同，不能解读成对不同姿态的广泛验证。复现实验脚本为 `validate_balance.py`。
- 详细 system prompt + GPT-6 direct + 踝关节辅助：`seed2-direct-balance-1000steps-01`，20 秒 / 1000 步全部执行、26 次 move_joints。第 137 步骨盆低于 0.35m，后续没有恢复到 0.6m；未进球，未证明稳定行走。模型收到可写代码生成轨迹的说明，但该回合没有执行 shell/Python 生成步态。不能将简单静态站立辅助描述成可靠行走控制器。
- 新 prompt 写明关节分组、坐标/单位、控制频率、球与目标观测、COM/足地接触、历史、代码辅助计算和机器人接口边界。两种原模式保留，平衡辅助显式 opt-in，官方套件默认时长仍为 300 步。
- 最新控制映射/模式隔离回归测试 4 passed。之前全套 92 passed 记录见下；没有将它冒充本次全套重跑。

## 初次接入验证

独立镜像 `world/humanoid-soccer:mujoco3.3.1` 已构建。RTX 5090 / 595.80 下 MuJoCo EGL 渲染可用，不依赖 Isaac Sim。官方源码与 World/codex 保持 clean。

官方 MuJoCo run_trial 默认出生分布，seed=7；每回合 6 秒 / 300 控制步（50 Hz），原版进球判据；全部完成，连续视频各 300 帧。不是完整论文指标复现，也不是 Isaac Lab 训练评测。

| 回合 | 模型工具调用 | 官方进球 | 解读 |
|---|---|---|---|
| baseline-seed7-01 | 无 GPT，原 PAiD ONNX | true | 底层原版踢球基线 |
| gpt6-hybrid-seed7-01 | GPT-6 (`gpt-6-astra`) review_action 8 次，全部 accept | true | GPT 审核原动作；没有实际触发修改，不应归为 GPT 独立学会踢球 |
| gpt6-direct-seed7-01 | GPT-6 (`gpt-6-astra`) move_joints 17 次 | false | 直接关节控制，跌倒后持续尝试恢复直至用完预算 |

基线与混合模式的 qpos/qvel/torque/time 全轨迹逐元素一致（最大绝对误差 0）；说明 accept 路径没有改变原版动作。modify 关节偏移、关节限幅、动作顺序映射与模式隔离已有单元测试，当前真实模型回合没有选择修改。直接模式不调用 ONNX.step；ONNX 元数据仅用于统一 PD 参数。两个 GPT 回合都由本地源码 Codex app-server 运行，无直接模型 API 客户端。

模型可见原版状态观测中的球/目标相对向量与增加的观察相机，明确不是纯视觉协议。完整状态轨迹只供评测后分析，不给 agent 源码/原始轨迹文件权限。球进门按 upstream goal_crossed 计算，不以 GPT 声称完成计分。

源码 Codex：8ae55c863db26d417e83390c5854f1144114276b。二进制 SHA-256：6ca8992ea4cda34050a14ca409239e4231a13196bdd2357a632ba5b1606a2892。

本机证据根目录：`World/var/runs/docker/humanoid_soccer/`。发布快照不含视频、权重或本机凭据。World 回归测试目前 92 passed。

## 参考失败教训的第二轮代码控制

`seed2-coding-balance-1000steps-02`：显式 controller-notes + 相同 seed/20s/direct/ankle-com，GPT-6 生成 10 段反馈程序并完成 1000 步，第 521 步骨盆低于 0.35m，未稳定行走/进球。前 400 步主要标定，不能仅凭倒地时间延后断言走路能力改善。程序回调中位约 1.94ms、P95 2.13ms（本机实测，非硬实时保证）。`informed_by_prior_attempt=true` 标明使用先前教训。相关 15 项测试通过；上游环境和 Codex 源码均保持不变。
