# Harbor acceptance record / Harbor 验收记录

Snapshot: **2026-10-17**. This is an evidence ledger, not a model leaderboard.
[English instructions](HARBOR.md) · [中文说明](HARBOR.zh-CN.md).

**Generated tasks, working HTTP services, completed Harbor trials and successful
physical solutions are separate claims.** 生成任务包、HTTP 接口跑通、Harbor trial
跑完、物理任务成功，分别记录，不能互相替代。

## Task coverage / 题库覆盖

- The registered set contains **84 tasks**. The exporter covers all 84 standard
  task packages, with Harbor 0.24.0 structure validation tracked in the catalog.
- This is **schema/export coverage**, not 84 successful runtime or model evaluations.
- `CountertopCleanup` inherits the existing World **600-step** budget from the
  legacy formal script. The installed upstream registry does not supply a
  verified native horizon for this identifier; `official_horizon: null` and
  `horizon_source` must retain that distinction. It is not labelled an official
  600-step benchmark limit.
- `tasks/catalog.json` is the machine-readable migration inventory.
  Runtime verification must be updated from evidence, not directory existence.

当前 84 题均可按标准格式导出；这不代表 84 道题已成功运行。
`CountertopCleanup` 使用旧正式脚本已有的 World 600 步配置，并明确记录官方
horizon 未知，不能把它宣称为官方原生预算。

## Physical/service evidence / 原生服务证据

All entries below use **no model**. They establish only the named infrastructure
properties. A neutral-action probe is not a task-solving reference.

| Benchmark / task | Execution | Actual / budget | Video evidence | Interpretation and local record |
|---|---|---:|---|---|
| RoboCasa / CloseDrawer | Harbor installed probe | 450 / 450 | 451 frames, 20 FPS | Full-budget transport run; no task success. `outputs/harbor/close-drawer-container-probe-01` |
| RoboLab / RubiksCubeLeftOfBowlTask | Harbor installed probe, absolute IK | 10 / 450 | 11 frames, 15 FPS; decoded | Query and request idempotence, MCP cameras, collection and separate verifier passed. Voluntary `agent_stopped`: the host audit classifies this as diagnostic, outside model scores; not a completed solving attempt. `outputs/harbor/robolab-probe01` |
| HumanoidSoccer / play-soccer | Harbor installed probe, direct joints + declared ankle assist | 1000 / 1000 | 1001 frames, 50 FPS; decoded | Native `run_trial` returned, authoritative success false, budget exhausted; zero Harbor exceptions. `outputs/harbor/soccer-probe01` |
| Digit / digit_walk_hand_tracking | Harbor installed probe | 10 / 700 | 11 frames, 50 FPS | Agent/API, idempotence and artifact collection worked. Initial verifier version errored; revised verifier classifies early stop as diagnostic, not a score. `outputs/harbor/digit-native-probe01` |
| ReflexBench / cup_ball_catching | Harbor installed probe, full native budget allowed | 30 / 100 | 31 frames, 25 FPS; decoded | Native failure ended the episode before its horizon. Separate verifier accepted reward 0, preserving original failure semantics. `outputs/harbor/reflex-native-budget-probe02`; earlier 10-step `reflex-native-probe01` remains an early-stop diagnostic. |
| AI-CPS / moving-ball catch (22) | Docker HTTP service; not yet a Harbor job acceptance | 298 / 300 | 299 frames, 60 FPS; decoded | Native done; actual RTAMT scorer returns success false, robustness -0.5946; valid failure. Earlier 3-step run remains diagnostic. `outputs/harbor-service-probes/ai-cps-22-full` |
| BEHAVIOR / clean_up_your_desk | Docker HTTP service; not yet a Harbor job acceptance | 3 / 2000 | 4 frames, 30 FPS; decoded | Gripper action, 12 RGB/depth history images, query/retry idempotence and finalization passed; agent-stopped diagnostic, no reward. `outputs/harbor-service-probes/behavior-desk-02` |
| RoboDojo / deposit_coin | Harbor installed macro probe | 1 / 300 | 3 cameras, 2 frames each, 25 FPS; decoded | Native waypoint executed; query/idempotence and independent grader checks passed. Early-stop diagnostic, no reward. `outputs/harbor/robodojo-deposit-macro-probe02` |
| WheeledLab / rw-reverse-bay | Docker HTTP service; not yet a Harbor job acceptance | 3 / 2000 | camera/onboard/rear: 4 frames each, 50 FPS; decoded | Query/retry idempotence passed; voluntary stop, no evaluation score. `outputs/harbor-service-probes/wheeledlab-reverse-bay` |

表内全部是**无模型基础设施检查**。短步数 smoke 提前停止不应纳入正式 SR。
历史 trial 的报错日志不删除；修复后的离线重评分也不能冒充原 trial 当时已成功验收。

### Native semantics retained / 保留的原生语义

- RoboLab uses the official double reset and frozen `get_env_results()` outcome.
  Its upstream reset handling can reject a termination in the first two steps as
  an initial physics artifact. Raw flags are logged, but are not relabelled as
  success when the native runner has not frozen the episode.
- Soccer keeps the original complete trial loop, controller mappings, PD stepping
  and goal-crossing checker. Success is published after the full native horizon,
  not by interrupting the runner when a transient intermediate condition appears.
- The 6.0.1 compatibility profiles remain explicitly experimental. A transport
  smoke does not prove equivalence to an upstream Isaac version.

RoboLab 早期物理 artifact 的处理、足球完整原生循环均保留。Isaac 6.0.1 实验兼容版
没有因为接口跑通就变成官方物理等价版本。

## Pending evidence / 尚待完成

**Scoring layers:** the verifier checks simulator evidence and emits completion
reward (1 for confirmed completion, otherwise 0). Normal model return before
completion counts as 0; an unavailable native score remains `native_success: null`.
The host audit then uses Harbor's trusted agent configuration/context to exclude
infrastructure probes. Any execution exception, including `AgentTimeoutError`,
is invalid rather than task failure. Run probes separately; raw Harbor reward
averages do not implement these exclusions. Report invalid/missing attempt counts
and coverage; a full benchmark score is unavailable until coverage is complete.

评分分两层：物理 verifier 返回 completion reward；正式模型正常返回但未完成算 0，
未运行原生评分器时 native_success 仍为空。host audit 依据 Harbor 配置及 host metadata
排除 probe，执行异常记 invalid，不能悄悄从统计中删除。旧记录仍保留原 verifier 的
诊断/缺 reward 结果，不能用新规则声称旧 trial 当时已经通过验收。

| Item | Status at snapshot | Required next evidence |
|---|---|---|
| OmniDrones native service | Asset dependency missing: `Assets/omnidrones/isaac41-ground/default_environment.usd` | Restore the correct asset, rerun the real service; no success claim |
| Remaining exported task variants | Structure accepted; individual scene runtime not established here | Per-task loading/stepping/video/checker checks |
| Containerized Codex model trial | Completed: DeepSeek via source-built Codex, 450/450 steps, native success false, zero Harbor exceptions | `outputs/harbor/close-drawer-container-codex-03/audit.json`: valid failure; 451 frames at 20 FPS decoded; request images at most 15; observable ATIF exported |
| Containerized Claude Code model trial | Completed: DeepSeek via Claude Code, 450/450 steps, native success false, zero Harbor exceptions | `outputs/harbor/close-drawer-container-claude-01/audit.json`: valid failure; 451 frames at 20 FPS fully decoded; observable ATIF exported |

Earlier Codex container setup attempts `...codex-01` and `...codex-02` produced
errors and remain archived. A verifier failure reward written after agent failure
must not be presented as a valid model attempt. Use the latest job and inspect
both agent exceptions and verifier details.

Codex 与 Claude Code 模型 trial 均已完成原生 450 步，任务未成功，运行及视频验收通过。早期容器/运行时错误保留，
统计时需检查 agent 异常，不能只看 reward 文件。

## Backup and rollback / 备份与回退

Before Harbor changes, the local checkout was archived at:

```text
../backups/robotworld-pre-harbor-20261017-010012/robotworld.tar.gz
SHA256: 961283bcbffd96739c05e047ac33d55fde550ec0ee426e72a3286d7d2488224a
Git HEAD recorded: 758022ec3b94a2c62c6deaab96701002a63b368b
Archive size: 484831395 bytes
```

This path is relative to the local `World/robotworld` checkout; it is not a file
shipped on GitHub. The same backup directory contains `SHA256SUMS`,
`manifest.json` and `RESTORE.txt`. Verify the checksum and follow `RESTORE.txt`,
restoring to a separate directory before replacing working code.

The archive excludes external Assets, external Codex source/builds, Python
virtual environments, private credentials and Docker images. Those dependencies
must remain available separately. This is a code rollback snapshot, not a full
machine/container backup.

备份是代码快照，外部资产、Codex 构建、虚拟环境、私有凭据和 Docker 镜像未重复打包。
请先校验哈希并恢复到独立目录，避免覆盖现有运行记录。

## Clean-checkout requirements / 新机器要求

The accepted local runs reuse provisioned dependencies. A new machine still
needs Docker GPU support, benchmark images, licensed assets, and the selected
agent runtime. Codex requires a verified source build and provider configuration;
Claude Code requires its pinned official CLI or a prebuilt agent image. Follow
[deployment](DEPLOYMENT.md) and [agent setup](../agents/README.md).

Current Compose packages reference `WORLD_ROOT` and `WORLD_ASSETS`; remote Harbor
workers must provision those contents locally. A task directory alone is not a
self-contained simulator/asset distribution. Native agent traces are retained.
`agents/trajectory.py` provides a limited, Harbor-schema-validated ATIF export of
observable messages and tool exchanges. It is not a lossless capture of private
reasoning, internal compaction, all runtime events or usage metrics. Real archived
Codex and Claude traces converted successfully: respectively 75/75 and 80/80
robot tool calls/results, with 504 and 522 image assets. Five converter tests pass.
Inspect exporter metadata before using these records for training.

本机验证复用了已安装依赖。新机器仍需镜像、授权资产及 agent runtime；远程节点要准备
自己的 `WORLD_ROOT` / `WORLD_ASSETS` 内容。原生轨迹保留，另有有限 ATIF 导出；
已用真实 Codex/Claude 记录及 Harbor schema 验证，但不包含隐藏推理、完整内部压缩过程
或推断出的 token/费用指标，不能宣称无损转换。

## Claude request-window attempt 02 / Claude 请求窗口第二轮

`outputs/harbor/close-drawer-container-claude-window-02` ended at **0/450 steps**.
The locally installed LiteLLM converter rejected `reasoning.effort` for the model
registry entry before any physical action. Harbor recorded
`NonZeroAgentExitCodeError`; host audit correctly reports **invalid**, not model
failure, even though the physical completion verifier wrote 0. The new Messages
window observed three real requests, all with zero images; this does **not**
validate image history clipping under load. Automatic post-sync ATIF export ran
and its two-step output passes the Harbor schema. The synthetic API error text is
an error record, not a model decision. A converter regression test now prevents
its `<synthetic>` marker from overwriting the configured provider model identity.

第二轮因本地协议转换器拒绝 reasoning 参数而在零步退出，已记 invalid，不能算模型
任务失败。自动 ATIF hook 已实际运行，但零图请求不能证明历史图像裁剪已完成线上验收。
旧 Claude 成功轮没有新窗口的 wire 证据，也不能替代这项验收。
