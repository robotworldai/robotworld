# Harbor 接入验收审计

> 这是 2026-10-19 的历史审计快照，不代表当前功能状态。Shell、可配置代码控制、原生工具反馈和 system 指令等已在后续修复。当前验收进度见 [迁移核验](HARBOR_MIGRATION_PARITY.zh-CN.md)。历史问题保留用于追溯，不删除原始结果。

审计日期：2026-10-19。对象：`World/robotworld` 当前工作区，而非已发布 commit。本轮检查代码、任务文件、已有运行产物，并执行离线测试；未修改运行配置、评分器或后台任务。

## 结论

目前是**已经实际使用 Harbor 运行的 RobotWorld 定制集成**，不是只有 Harbor 外观的目录。但尚未达到“84 题完整验收、旧功能全部保留、任意 Harbor agent 和云环境无痛替换”的程度。

任务格式、框架运行、任务可运行、任务可完成是四个不同层次，不能混为一谈。

## 按工程收益核对

| 能力 | 状态 | 已有证据与剩余边界 |
|---|---|---|
| 公共任务格式 | 已实现 | 84 个任务目录都有 instruction.md、task.toml、environment/docker-compose.yaml、tests/test.sh；84 个 TOML 均可解析。任务目录用任务名，来源保留在元信息中。 |
| 真正由 Harbor 执行 | 已实测，覆盖不完整 | job 配置使用 Harbor tasks/agents/n_attempts，已有 trial/config/result/verifier/artifacts。不是仅靠旧脚本假装 Harbor。 |
| 环境与 agent 解耦 | 已实现并有局部实测 | agent main 容器、simulator 服务、独立 verifier；经 MCP 和 episode service 交互。尚未做完整运行时权限穿透验收。 |
| 通用机器人接口 | 已实现 | describe、observe、status、execute、receipt；execute 提交有序原生动作命令，支持请求标识、回执、过期观测检查。agent 不需要直接访问 simulator Python 对象。 |
| Codex / Claude Code | 两种定制 agent 已实测 | 两者均在容器中运行，使用 agents.robotworld 中的 BaseInstalledAgent 子类；CloseDrawer 双 harness 已产生动作、视频和评分。 |
| 任意 Harbor agent 直接替换 | 未完成验收 | 目前不是仅改 `-a` 就已保证兼容。新 agent 仍需处理 MCP、多模态观测、继续交互、模型协议及轨迹收集。不能将两个定制 agent 的成功启动推广到全部内置 agent。 |
| Shell 和离线分析 | 未保留在当前专用入口 | 后续逐行核查纠正：Codex 设置 shell_tool=False、code_mode=False；Claude Code 使用 --tools 空列表。容器存在 shell 不等于模型可以调用 shell。 |
| 反馈代码控制 | **未保留** | Harbor catalog、agent metadata 和多个 service 明确固定 code_control=False；部分入口直接过滤 coding_control 工具。这不是默认关闭后可随任务打开，尚有功能缺口。 |
| 环境复现 | 部分完成 | 有 Compose、镜像和资产恢复机制。但任务依赖 WORLD_ROOT/WORLD_ASSETS 外部挂载、预先准备的镜像与源码；不能只复制单个任务目录到空机器运行。尚无干净机器全量验收。 |
| 批量执行和重复次数 | 已实现 | 全题 job、n_attempts、42/42 随机分配 harness 的脚本都有；默认 GPU 串行。混合分配不是同题双 harness 对照实验。 |
| 种子、恢复、云端调度 | 部分完成或未验证 | 重复 attempt 不自动更换模拟器 seed。当前故障恢复还使用 var 下私有脚本；未交付完整公共恢复流程。多 GPU、远程云沙箱未验收。 |
| Harbor 分数输出 | 已实现，可靠性仍待全面验收 | verifier 输出 reward.txt、details.json，宿主检查结果与仿真记录一致性；可排除显式执行异常。但正常返回的 agent_stopped 会计为失败，静默工具转换问题可能漏过基础设施分类。 |
| 均值、方差、覆盖率 | 已实现基础统计 | check_harbor_results.py 汇总有效运行成功率、总体方差、分任务统计与覆盖率。不应把仅已发现的 trial 目录覆盖率当成全部 84 题计划覆盖率；缺失计划任务还需对照 catalog/job。 |
| 视频和执行记录 | 已实测，覆盖不完整 | 有 simulator 视频、episode、agent 原始事件和 ATIF trajectory。ATIF 明确为有损导出，不保证完整 usage、压缩事件等信息；原始记录仍需保留。 |
| 参考解与可解性 | **未完成** | 84 个任务目录中 solution/solve.sh 为 0。Harbor 参考解可选，因此不等于格式不合格；但不能用 oracle 批量证明任务可解。旧参考轨迹尚未等同于 Harbor 可回放参考解。 |
| 迁移前后语义一致 | 未完整验证 | 保留原生动作与 checker 的设计，不等于逐题验证通过。代码控制、观测历史、继续交互、步数预算、仿真兼容版本均需对照检查。RoboDojo 原子宏动作可能超出外加控制步上限。 |
| 文档与造题指南 | 已有，需收口 | 英中 README、Harbor 指南、模板均存在；历史验收文档未完全反映最新运行状态，应避免把计划写成已完成。 |

## 实测证据如何解读

- `outputs/harbor/close-drawer-sub2api-both-20261018`：Codex、Claude Code 各一条，无 Harbor 异常，均实际推进到 400 步，均未成功。证明这条基础链路可执行，不证明任务可解性已验收。
- `all84-mixed400-sub2api-20261018-codex`：42 条结束，其中 27 条带异常；其余 15 条不是“15 题成功”。
- 同批 Claude Code：42 条结束，42 条带异常。模型服务故障不能算成 42 个任务能力失败，也不能证明仿真均不可运行。
- 新账号续跑 `all84-sub2api-resume-20261019-claude-code`：本轮检查时已有 2 条结束，无异常，reward 均为 0。计数会随后台运行变化。
- 早期 DeepSeek/Codex 批次存在无异常但 0 控制步的记录。这说明“进程正常退出”不足以作为接入成功标准，仍要查有效动作、协议转换和退出原因。

不同模型、服务、版本、预算和种子的记录不能拼成一个正式成功率。当前证据不支持宣称全部 84 题通过 Harbor 运行验收。

## 本轮离线验证

使用现有测试环境运行以下测试，**71 项通过**：

```text
tests/test_episode_protocol.py
tests/test_harbor_agents.py
tests/test_harbor_verifier.py
tests/test_harbor_result_classification.py
tests/test_atif_export.py
environment/tests/test_harbor_export.py
```

这只证明已覆盖的协议、导出、评分和 agent 行为测试通过，不替代 GPU 仿真、干净部署或成功轨迹测试。

## 建议验收顺序

1. **恢复承诺的功能**：将 code_control 做成显式配置，保持默认关闭，并测试允许观测和动作权限边界；补同题开关对照。
2. **先辨认运行是否有效**：区分无动作模型收尾、工具格式丢失、环境初始化错误、正常原生失败和预算耗尽；不能仅凭 agent_stopped 分类。
3. **完成 84 题运行矩阵**：按计划逐题记录 reset、有效 step、图像、视频、终止、verifier 和产物；未运行与运行失败分开。400 步冒烟不等于原生完整回合。
4. **补可解性验证**：把真实 step 驱动参考程序接成 solution/solve.sh，运行 oracle，再做成功/失败边界检查；不能写 reward 伪造参考解。
5. **完成迁移一致性**：相同初始状态与动作回放比较新旧轨迹、步数和判据，记录不等价的实验兼容版本。
6. **公开交付与扩展**：干净机器从 README 部署；公共恢复命令；至少一个新增 agent、一个远程 GPU 环境验收；跨任务计划完整性统计。

其中 1–3 是当前接入优先项；4–5 决定 benchmark 可信度；6 决定外部用户是否真正能够无痛使用。

## 核心代码索引

- [任务生成器](../environment/harbor_tasks/export.py)
- [任务配置示例](../tasks/close-drawer/task.toml)
- [环境依赖示例](../tasks/close-drawer/environment/docker-compose.yaml)
- [两个定制 harness](../agents/robotworld.py)
- [MCP 接口](../environment/bridges/mcp.py)
- [评分和覆盖率检查](../scripts/check_harbor_results.py)
- [批量 job](../configs/jobs/robotworld-codex.yaml)
- [中文使用指南](HARBOR.zh-CN.md)

官方对照：[任务格式与任务独立性](https://docs.harborframework.com/tasks/overview)、[预集成 agents](https://docs.harborframework.com/agents/pre-integrated-agents)。官方框架具有某项能力，不代表 RobotWorld 的 GPU、多容器和多模态集成已对该能力完成验收。
