# 在 Harbor 中使用 RobotWorld

推荐的 Harbor 内置 agent 流程见[首页快速开始](../README.zh-CN.md#标准-harbor-入口)。修改任务、接入 harness 和回归验收见[开发指南](HARBOR_DEVELOPMENT.zh-CN.md)。下文 probe 与源码适配器属于独立的诊断或旧配置，不是内置 agent 流程的前置依赖。

[Validation evidence and pending items / 验收证据与待办](HARBOR_VALIDATION.md).

[English](HARBOR.md) · [完整机器人接口](harness-protocol.md) · [安装说明](DEPLOYMENT.md)

**沿用 Harbor 的 task、agent、job、trial；RobotWorld 提供物理环境和动作接口。** 更换模型或 agent，不应改变任务物理、观测权限、初始状态分布或评分标准。

本集成针对 **Harbor 0.24.0**。全部 84 道选题已有标准任务包，但目录/结构校验不等于
每个场景都已真实运行。实际证据以[验收记录](HARBOR_VALIDATION.md)为准。

**Task** 定义环境和目标；**agent** 运行 harness，与模型和环境交互；**trial** 是某个
agent 在一道题上的一次尝试；**job** 负责调度并收集一组 trial。造题修改 `tasks/`，
接 harness 修改 `agents/`，选题、模型与重复次数修改 `configs/jobs/`。
`n_attempts` 重复当前任务配置，不会自动随机更换 seed。

## 各部分在哪里运行

| Harbor 部分 | 包含内容 | 权限 |
|---|---|---|
| 主机 scheduler | job 配置、trial 创建和结果收集 | 启停容器 |
| `main` 容器 | installed agent、模型客户端、HTTP/MCP 客户端 | 读公开观测、提交动作 |
| `simulator` 容器 | 原生 benchmark、资产、episode 服务 | 推进仿真、录视频、检查状态 |
| 独立 verifier | 评分代码与收集到的仿真产物 | 输出 reward 和详细结果 |

使用远程模型 API 时，agent 容器不需要 GPU。不同 benchmark 继续复用各自的仿真镜像，不必为每个模型重新构建仿真。agent 不应挂载 Docker socket、仿真源码、资产或可写的仿真结果目录。模型密钥只提供给 agent；环境接口使用独立的 episode token。

## 安装与启动

先按[部署指南](DEPLOYMENT.md)准备 Docker GPU 支持、对应镜像和授权资产，再在独立 Python 环境安装 Harbor：

```bash
python -m pip install 'harbor==0.24.0'
# 以下在仓库根目录执行。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_ROOT="$PWD"
# 验证资产路径，并在 Docker 以只读方式挂载仓库前创建挂载点。
WORLD_ASSETS="$(python scripts/check_harbor_assets.py "${WORLD_ASSETS:-$PWD/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
export WORLD_NVIDIA_RUNTIME="$(python scripts/harbor_gpu_runtime.py)"
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
harbor run --path tasks/close-drawer --agent-import-path agents.robotworld:RobotWorldProbeAgent \
  --n-concurrent 1 --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

脚本 probe 执行完整的 450 个原生控制步，输出完整视频及评分结果；它仅验证基础设施，不证明模型能力或任务可解。若换成 `--agent nop`，环境未执行动作就结束，completion verifier 返回 0，但 host audit 会把该 probe 排除于模型成绩。probe 必须单独建 job，不能直接混用 Harbor 原始 reward 聚合。安装 Harbor 不会自动装好所有 agent、模拟器和资产。

使用自己的 installed agent 无需接入另一套调度器：

```bash
harbor run --path tasks/close-drawer \
  --agent-import-path your_package.agent:YourInstalledAgent --model your-model-id \
  --n-concurrent 1 --n-attempts 3 --jobs-dir "$PWD/outputs/harbor"
```

先实现 Harbor installed-agent 接口，再让它在 `main` 内启动自己的 harness。`harbor agent schema <agent>` 查看参数，`--ak key=value` 传 agent 专用选项。仓库提供的实现见 [agents/README.md](../agents/README.md)：`agents.robotworld:RobotWorldCodexAgent`、`agents.robotworld:RobotWorldClaudeCodeAgent` 和无模型的 `RobotWorldProbeAgent`。Harbor 内置同名 agent 不保证与这些实现采用相同的续聊、图片裁剪和工具权限。

### 可选旧配置：仓库提供的 installed agent

Codex 复用经过核验的源码构建，再准备容器内可重定位的 runtime 包：

```bash
export WORLD_CODEX_SOURCE=/absolute/path/to/verified/codex/source
python scripts/prepare_harbor_codex.py \
  --provider-config /private/path/provider.json --output var/harbor-runtimes/codex-provider
export WORLD_HARBOR_CODEX_RUNTIME="$PWD/var/harbor-runtimes/codex-provider"
# 先从私有文件加载 provider 要求的密钥环境变量。
harbor run --path tasks/close-drawer \
  --agent-import-path agents.robotworld:RobotWorldCodexAgent --model your-model-id \
  --n-concurrent 1 --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

Claude Code 使用固定版本官方 CLI（当前 2.1.292），以及容器内能访问的 Messages 地址：

```bash
export WORLD_HARBOR_CLAUDE_BINARY=/absolute/path/to/claude
# 私下加载 ANTHROPIC_BASE_URL 和 ANTHROPIC_API_KEY。
harbor run --path tasks/close-drawer \
  --agent-import-path agents.robotworld:RobotWorldClaudeCodeAgent --model your-model-id \
  --n-concurrent 1 --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

适配器启动进程时注入选定凭据，不把真实 key 写进 `--agent-env` 或 job 文件，模拟器不接收模型密钥。无模型 CloseDrawer 全步数接口检查可选 `agents.robotworld:RobotWorldProbeAgent`；这个脚本不解题，不能作为模型成绩。

也可以直接使用标准 job YAML：

```yaml
# job.yaml，在仓库根目录运行。
jobs_dir: outputs/harbor
n_attempts: 3
n_concurrent_trials: 1
agents:
  - import_path: agents.robotworld:RobotWorldProbeAgent
tasks:
  - path: tasks/close-drawer
```

```bash
harbor run -c job.yaml --jobs-dir "$PWD/outputs/harbor"
```

Harbor 0.24.0 的 Compose 产物复制可能把相对路径按 task 目录解析，请使用绝对 `--jobs-dir`。

远程训练/评测节点仍需准备自己的镜像与资产路径；本地 bind mount 不会自动成为远程数据包。

## 逐题 100 步冒烟检查

这是不调用模型的部署诊断，与正式模型评测分开。顺序是先恢复固定版本源码，再准备
资产，最后确保任务包所指定的镜像全部存在：

```bash
# 主机与 Python 安装步骤见 DEPLOYMENT.md。
bash scripts/setup_handoff.sh sources
bash scripts/setup_handoff.sh assets
# 已有匹配资产时，只恢复运行路径，无需重新下载：
# python scripts/restore_assets.py --apply
# 按 PREBUILT_IMAGES.md 导入镜像；缺少的任务 profile 需另外构建。
python -m pip install 'harbor==0.24.0'
export WORLD_ROOT="$PWD"
# 验证资产路径，并在 Docker 以只读方式挂载仓库前创建挂载点。
WORLD_ASSETS="$(python scripts/check_harbor_assets.py "${WORLD_ASSETS:-$PWD/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_NVIDIA_RUNTIME="$(python scripts/harbor_gpu_runtime.py)"
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
harbor run --config configs/jobs/robotworld-smoke100.yaml \
  --jobs-dir "$PWD/outputs/harbor"
```

通用资产包不包含受限 BEHAVIOR 数据，每位使用者仍需按 [ASSETS.md](ASSETS.md)
接受官方许可并单独下载。导入镜像只恢复包中列出的 tag，不代表所有旧版本或实验
profile 都已安装；主机仍需兼容的 GPU 驱动和 Docker GPU 支持。本地已有的匹配安装
可以直接复用。

当 Docker 的默认运行时为 `runc` 且已注册 NVIDIA 运行时时，启动脚本会为 RoboDojo
选择该 NVIDIA 运行时。可通过 `WORLD_NVIDIA_RUNTIME` 指定运行时，或设为空字符串以使用 Docker 的默认值。

上述变量只配置环境连接：`WORLD_ROOT` 和 `WORLD_ASSETS` 选择容器挂载，
`PYTHONPATH` 让 Harbor 导入 installed-agent 模块，`WORLD_EPISODE_TOKEN` 用于
环境 API 鉴权。task、agent、job、trial 和 verifier 均沿用 Harbor 原生约定。
100 步停止条件属于 `agents.smoke100:RobotWorldSmokeAgent`，不会改写 `task.toml`、
原生题目预算或成功判据。

`bash scripts/run_harbor_smoke.sh` 仅是上述原生命令的便利包装。
包装脚本自动定位仓库，`WORLD_ASSETS` 默认指向仓库的 `Assets/`，缺少 episode token 时
自动生成；结果默认写到仓库的绝对路径 `outputs/harbor/`。`HARBOR` 可以指定已安装
CLI 的绝对路径，`WORLD_ASSETS` 可以指向外部已有资产。所有脚本参数透传给
`harbor run`，例如：

```bash
HARBOR=/absolute/path/to/venv/bin/harbor \
WORLD_ASSETS=/absolute/path/to/Assets \
bash scripts/run_harbor_smoke.sh --job-name smoke100-local-01
```

每次新批次使用新的 job 名称。默认全部 84 题、每题一次、同时运行一个 trial；
可修改配置的 `tasks` 选取子集，按主机容量调整 `n_concurrent_trials`。
无需模型 API key。执行器为 `agents.smoke100:RobotWorldSmokeAgent`。

配置为 `configs/jobs/robotworld-smoke100.yaml`，每题一次 trial。脚本 agent 通过与
模型相同的公开 episode API 发出合法动作。100 步指原生**控制步**，不是模型轮数、
工具次数、视频帧数或物理子步。原任务预算、成功/失败和终止条件优先：若提前终止，
记录实际步数与原因，不屏蔽原生失败，也不在同一 trial 内 reset 凑足 100 步。
宏动作接口按真正执行的步数记录，不能把一个命令直接算作一步。

检查覆盖场景加载/reset、动作接收执行、观测传输、录制收尾和 verifier 执行。
中性或脚本动作不必解题；原生任务失败与基础设施错误分开记录。视频可以播放也不能
单独证明成功判据正确，正反例仍需专项验证。这些 smoke 不计入模型成功率。

Harbor 每次生成一个 job，里面每道题的一次尝试是一个 trial：

```text
outputs/harbor/<job>/
  config.json                 # 实际 job 配置
  result.json                 # job 汇总，包括 trial 异常
  <trial>/
    result.json               # 本轮结果和异常
    agent/robotworld/          # smoke100 的 summary.json 和 receipts.jsonl
    artifacts/simulation/     # 仿真视频、原生动作与状态记录
    verifier/                 # reward 和验收明细
```

各 bench 的视频名称、子目录不同，在 `<trial>/artifacts/simulation/` 下查看 `.mp4`。
中断或报错可能留下不完整产物，需保留异常，不能记为完成了 100 步检查。使用
`scripts/check_harbor_results.py` 对整个 job 做独立证据审计；它会把 probe 排除出
模型分数，不能把 probe 的 reward 当作模型表现。

正式评测改用带模型和凭据的 installed agent，再运行
`configs/jobs/robotworld-codex.yaml` 或自己的标准 job。原生预算和已声明的 World
任务配置仍保留在 task 中，不会被这次诊断改为 100 步。

## 两种 agent 的 400 步诊断

全部任务直接放在 `tasks/<task-name>/`，`tasks/catalog.json` 记录来源 benchmark 与
原生预算，不再按 benchmark 嵌套，也不另放一套 pilot 任务。

先按固定随机种子分配 84 道题，生成两个标准 Harbor job：

```bash
python scripts/run_harbor_mixed.py --prepare-only \
  --model deepseek-v4.1-flash --steps 400 --seed 17 --name mixed400-01
```

生成文件位于 `outputs/harbor-plans/mixed400-01/`：`assignment.json`、`codex.yaml`
和 `claude-code.yaml`。随机打乱后，42 题交给 Codex，42 题交给 Claude Code，每题
一次。这里 seed 只决定**题目分配**，不改变模拟器初始状态。两边使用相同模型 ID，
但每题只由一个 harness 执行，因此不是同题的配对对照实验。

按上文准备两个 installed agent，私下加载模型凭据，并设置常规挂载/API 环境变量，
然后直接运行原生 Harbor 命令：

```bash
export WORLD_CONTROL_STEP_LIMIT=400
harbor run --config outputs/harbor-plans/mixed400-01/codex.yaml \
  --jobs-dir "$PWD/outputs/harbor" --job-name mixed400-01-codex
harbor run --config outputs/harbor-plans/mixed400-01/claude-code.yaml \
  --jobs-dir "$PWD/outputs/harbor" --job-name mixed400-01-claude-code
unset WORLD_CONTROL_STEP_LIMIT
```

`WORLD_CONTROL_STEP_LIMIT` 是额外诊断上限，不是新的官方题目预算。最多执行
400 与原任务预算中较小的步数，保留原生提前终止和已有 checker。结果必须记录
这个上限，不能当作原预算下的正式分数；取消变量后，后续 job 恢复声明的任务预算。

也可以使用便利包装：

```bash
bash scripts/run_harbor_mixed.sh --steps 400 --model deepseek-v4.1-flash \
  --seed 17 --name mixed400-02
```

它生成相同配置，顺序执行两个 job，每次只运行一个 trial。`HARBOR` 可以指定 CLI，
`WORLD_ASSETS` 可以指向外部资产。计划目录另存 job 日志和 `launch-results.json`；
真正的视频、agent 轨迹和 verifier 结果仍在标准的
`outputs/harbor/<job>/<trial>/`。生成计划不代表已启动或跑通，实际证据见验收记录。

## 如何接入更多 harness

task 的 `task.toml` 声明 MCP 服务：

```toml
[[environment.mcp_servers]]
name = "robotworld"
transport = "stdio"
command = "python"
args = ["-m", "environment.bridges.mcp"]
```

支持 MCP 的 harness 在 `main` 里启动此服务，继承 `WORLD_EPISODE_URL` 和 `WORLD_EPISODE_TOKEN`。自定义 installed agent 必须把 task 的 MCP 配置交给自己的运行时；仅在 TOML 里声明并不能让一个忽略 MCP 的 agent 自动使用它。工具是 `describe`、`observe`、`status`、`execute`、`receipt`。

也可以不使用 MCP，直接调用 HTTP：

```python
from environment.clients.episode import EpisodeClient
client = EpisodeClient()
spec = client.request("describe")
observation = client.request("observe")
status = client.request("status")
```

`GET /v1/describe` 返回任务、动作 schema、单位、参考系和预算；`GET /v1/observe` 返回公开图像和状态。调用 `POST /v1/execute` 时带唯一 `request_id`、当前 `episode_id`、`observation_id` 和有序 commands。完整请求见[接口示例](harness-protocol.md#public-api)。网络超时后先查 `GET /v1/receipts/{request_id}`，不要换一个 ID 盲目重发动作。

chunk 长度由 agent 决定，仍受传输和原生动作参数限制。环境先验证整个 chunk，再顺序执行，按实际控制步计数，达到终止条件或预算立即停止。底盘、手臂同时运动应使用原生组合动作，不是两个并发写请求。查询观测不会推进物理时间。

RoboCasa 每个观测包包含当前帧及间隔两个控制步的四个历史帧，按相机分别返回。**这不等于自动删除对话里的旧图片。** agent 还需要自己的输出请求图片窗口，同时完整归档原始轨迹。模型文字宣称结束不能判成功；正常 turn 结束后的继续策略和交互上限要对所有模型一致。

离线 shell 可以处理公开图片、计算动作；环境内 `code_control` 是单独声明、默认关闭的受限能力，不是开放模拟器 shell。更换 harness 不能额外获得隐藏状态、重置或改写环境的权限。

## 模型与 API

模型和 agent 独立：DeepSeek 在 Claude Code 中运行，仍是 Claude Code harness。Codex 需要 Responses 协议；Claude Code 需要 Messages 协议。Chat Completions 地址不能仅靠换 model 字符串就自动兼容二者。需要时配置协议转换器，并验证文本、图片、工具调用和流式返回，详见[第三方 API 示例](HARBOR_VALIDATION.md)。

容器里的 `127.0.0.1` 指向容器自身。外部网关必须使用 `main` 能访问的地址。密钥由私有环境配置注入，不放 task、YAML、公开日志或命令行示例；不要把 ChatGPT 账号转换当成第三方 API 接入方式。

## 导出题目与造新题

```bash
python scripts/export_harbor_tasks.py --list
python scripts/export_harbor_tasks.py --bench robocasa \
  --tasks CloseDrawer,CoffeeSetupMug --seed 7 --output /tmp/robotworld-task-export
```

导出器只生成已有适配器支持的任务，另附迁移清单。尚未支持的旧入口应显式报错，不能伪造空壳任务或默认评分。当前可用范围以脚本 `--help` 和 catalog 为准。

新增 benchmark 时，先实现和验证 episode adapter，保留原生动作、reset 和终止语义，明确标记自定义评分条件。然后按标准 Harbor 目录组织：

```text
my-task/
  instruction.md                  # 公开目标和交互说明
  task.toml                       # 墙钟预算、MCP、产物收集
  environment/Dockerfile          # agent 侧任务依赖
  environment/docker-compose.yaml # main + simulator
  tests/test.sh                   # 调用独立评分器
  tests/grade.py                  # 校验仿真端产物
  solution/solve.sh               # 可选；仅放真正验证的参考解
```

[造题模板](../templates/harbor-task/README.md)有意不能直接运行，必须补好真实环境和判据。不要放“写 reward 文件”的假参考解。任务元信息/产物记录原生 task ID、seed、控制步上限、checker 版本和资产来源。

## 步数、重复和成绩

- 控制步、物理子步、模型调用次数、墙钟时间是不同单位。`task.toml` 的 agent timeout 是墙钟时间，控制步上限由仿真服务执行。
- CloseDrawer 原生上限 450 控制步，agent 墙钟预算 43200 秒；两者不能互相替代。
- `--n-attempts 3` 表示三个 trial，不会自动生成三个 seed。固定 seed 测的是同一初始化下的 agent 波动；环境变化应另导出明确种子的任务并记录。
- verifier 根据仿真端最终产物评分，不相信 agent 写的成功声明。有效成功 reward 1，有效未成功 reward 0；原生连续分数单独保留。
- 仿真异常、产物损坏/缺失、checker 不兼容必须记 error，不能当普通失败。还要检查 agent 自身异常，只有 reward 文件不代表运行有效。
- 同时报告请求 trial 数、有效数、无效数、有效 trial 成功率和覆盖率。若采用所有尝试为分母，需明确披露。有效 trial 少于两次时不能声称得到样本方差。

Harbor 的 job/trial 目录是最终结果来源：保留 agent 轨迹和文件、仿真视频和逐步动作、verifier reward/details 与运行配置。中断重跑从新 episode 开始，除非确实实现了经过验证的物理状态恢复；不要静默丢弃失败或仅看视频判成功。

## 新任务验收

至少检查启动/reset、单位与参考系、请求幂等、预算终止、完整视频、checker 正反例和产物隔离；先无模型测试，再模型 trial。持续时间条件根据实际控制频率换算秒数。分别记录“适配器实现”“冒烟通过”“模型跑完”“成功轨迹验证”，不能混为一个完成状态。

官方参考：[task 格式](https://docs.harborframework.com/tasks/overview)、[多容器](https://docs.harborframework.com/tasks/multi-container)、[custom agents](https://docs.harborframework.com/agents/custom-agents)、[独立评分](https://docs.harborframework.com/tasks/separate-verifier)。命令和配置字段对照本机 Harbor 0.24.0 源码核查。

## 可观测 ATIF 轨迹

自定义 installed agent 保留原始轨迹，并支持有限的 ATIF 导出：

```bash
python -m agents.trajectory --logs-dir /absolute/job/trial/agent --harness codex
# Claude Code 使用 --harness claude-code
```

输出 `trajectory.json` 通过 Harbor 轨迹 schema 验证；图片原字节解码后存入
`trajectory-media/`。消息、工具调用及已有回执来自真实事件，原始日志始终保留。
这不是无损转换：不重建隐藏推理、内部压缩及缺失的续聊 prompt，不推断 token、费用
或 reward。用于训练前应检查 `extra.limitations` 与来源索引；ATIF step 是事件序号，
不是物理控制步。

## 全部选题的标准 Job

准备对应环境与资产，并修改 `YOUR_MODEL_ID` 后，可用标准 job 调度全部 84 题：

```bash
harbor run --config configs/jobs/robotworld-codex.yaml --jobs-dir "$PWD/outputs/harbor"
```

直接修改文件中的 `n_attempts`（默认 3）、`n_concurrent_trials`（默认 1）、
`agents` 和 `tasks` 列表。默认共 252 次 trial，不是冒烟测试。seed 由导出的任务变体
决定，增加 attempts 不会自动换 seed。当前实际验证及评分边界见[验收记录](HARBOR_VALIDATION.md)。

### 完成分数与审计

正式模型正常返回但目标未完成，completion reward 为 0。原生评分器尚未执行时，
`native_success` 保留空值，不伪造原生分数。API、协议、执行异常（包括 Harbor
`AgentTimeoutError`）记 invalid，并必须报告无效次数及覆盖率。

```bash
python scripts/check_harbor_results.py "$PWD/outputs/harbor/JOB_NAME" --output audit.json
```

审计根据 host 侧 Harbor 配置及 agent context 排除 probe，核验物理证据；仍有缺失或
无效正式尝试时不输出完整 task-macro 分数。probe 应独立建 job，不能把 Harbor 原始
reward 均值直接当作审计后的模型成绩。原生分数缺失与正常返回未完成得到 completion 0
是两件事。

### 环境负责结束与判定

仿真服务是 episode 完成状态的唯一来源：每次动作执行后接收配置好的环境/checker 返回，决定继续或结束，并把最终状态保存到 `episode.json`。环境不等待 Harbor verifier 返回。Harbor 在执行结束、收集产物后才运行 verifier。

verifier 仅检查任务身份及最终结果格式，将 `success` 转为1或0分。不读取辅助的 `result.json`、`native-result(s).json`，不重新计算物理判据，也不根据视频质量改变得分；这些产物仅供独立诊断。最终结果缺失、尚未结束、格式错误或基础设施错误属于无效运行，不记作任务失败。agent 提前返回或预算耗尽且未确认成功，记0分。诊断复验另存，不静默覆盖旧 trial。
