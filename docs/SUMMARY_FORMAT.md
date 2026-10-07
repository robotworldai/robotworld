# 每题评测 summary 包含什么

多次 rollout 统一评测的每题 summary 在：

```text
outputs/<benchmark>/<task-name>/summaries/<batch>.json
```

同一 benchmark 的总汇总在 `outputs/<benchmark>/summaries/<batch>.json`，对应 CSV 保存每题各指标的统计行。不同 batch、模型和预算不混合。当前冒烟每题仅一次 rollout，不能当作充分的模型成功率评测。

| 字段 | 含义 |
|---|---|
| `benchmark`, `task`, `task_title`, `legacy_task_id` | 项目、具体任务名、中文说明（已登记时）、历史编号（无则 null） |
| `configuration` | 模型、rollout 次数、任务筛选、预算设置、code control 请求开关、种子、runtime、墙钟超时、批次等；认证目录不写入配置 |
| `requested` / `completed` | 计划次数 / 正常取得原生结果的次数；completed 包括原生任务失败 |
| `infrastructure_errors`, `interrupted`, `not_run` | 运行时错误、中断、未运行数量，均不作为普通任务失败进入 SR |
| `unscored_success` | 有有效原生结果，但上游没有提供布尔成功值的次数 |
| `artifact_warnings` | 有文件缺失等归档警告的 rollout 数，不是警告条目总数 |
| `success_rate` | 所选评分版本的有效布尔成功结果的有效样本数 n、均值、总体方差、样本方差 |
| `scores` | 每个有限数值指标各自的 n、均值、总体方差、样本方差；例如 reward、return、robustness、q_score、完成比例、运动指标，取决于 benchmark 实际提供什么 |
| `runs` | 每次 rollout 的完整调度记录：状态、模型、实际/原生步数上限及来源、运行配置、工具开关实际是否生效、种子、开始/结束时间、运行路径和命令、结果或错误信息 |

`runs[i].result` 包含：

- `success`：true、false 或 null。null 表示没有完整有效二值裁定（如考程未结束或字段缺失），不能当成失败或成功。
- `scores`：本次原始标量分数；不是平均后的值。
- `result_file`：相对本 run 的 artifacts 目录的原生结果文件。
- `success_definition`：原生结果有提供时保存，否则可能为 null。
- `stop_reason`、`control_steps`：原生结果能提取到的停止原因和实际控制步数；缺少对应字段时可能为 null，不凭日志长度猜造。

`runs[i].previous_attempts` 保留中断重跑的旧目录与状态。汇总每个逻辑 rollout 只计最新一次，不重复计旧的部分尝试。切换任务名不会改分数或实际动作。

## 均值与方差的定义

- SR 是有效成功/失败结果的 0/1 均值：成功次数 / 有明确布尔结果的有效次数。
- `variance` 是总体方差（ddof=0），SR 对应 p(1-p)；`sample_variance` 是样本方差（ddof=1）。不是多个独立实验批次的成功率方差，也不是均值的估计方差。
- 只有一个有效样本时总体方差为 0，样本方差为 null；没有有效样本时均值和方差为 null。
- 各 score 单独使用它实际存在的有效样本。完整数组轨迹不平均塞进 summary；不同任务的 reward 标度也不混合平均。

## 视频、过程代码和失败分析

summary 用 `runs[i].path` 指向完整运行目录；视频、完整及无图像版轨迹、程序和 workspace 分别见该目录的 `videos/`、`events/`、`programs/`、`agent-workspace/`。文件索引见 `artifact-index.json`，未产生的文件不会凭空补齐。

当前 summary **不是自动撰写的失败原因分析报告**：它保存原始结果、停止原因、运行错误以及证据路径，不自动判定“为何没抓住”或“为何摔倒”。需要结合 tools/environment/codex 轨迹与视频进一步分析。原始环境若另有 `artifacts/summary.json`，那是该环境自己的格式，统一统计以上述 `summaries/<batch>.json` 为准。

冒烟链路审计单独位于每个 run 的 `smoke-audit.json`，跨项目表在 `outputs/_smoke/<batch>/README.md`：检查实际动作调用、物理推进、提示词和工具配置、日志、视频及索引。**smoke_passed=true 不等于任务成功**。

跨项目单一总分的候选方案见 [UNIFIED_SCORE.md](UNIFIED_SCORE.md)。现有 summary 仍保留原始 SR 和各原生 score，不直接平均量纲不同的分数。

## 交接版bench汇总

`overall`包含按题等权SR的均值和题间方差、pooled有效rollout的0/1均值方差、覆盖数及完整性。缺少计划内有效二值结果时，`full_benchmark_success_rate=null`。`<batch>-runs.csv`逐行列出每次rollout，不把不同任务的原score混平均。
