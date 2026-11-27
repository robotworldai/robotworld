# 使用 Harbor 内置 agent

[English](HARBOR_BUILTIN_AGENTS.md)

现在提供独立的标准入口：`name: codex`、`name: claude-code`。这些配置使用 Harbor 内置 agent，不导入我们的 Codex／Claude 模型包装器，也不要求源码构建的 harness 包。

## 分别选择任务、agent 和 model

生成的是标准 Harbor YAML，不会额外引入自定义 agent 循环：

```bash
python scripts/make_harbor_job.py --agent codex --model openai/gpt-6-astra \
  --agent-version 0.164.0 --task close-drawer --attempts 3 --concurrency 1 \
  --output configs/jobs/my-experiment.yaml
bash scripts/harbor_run.sh --config configs/jobs/my-experiment.yaml
```

`--task` 可重复指定，`--task all` 选择全部 84 题；`--attempts` 是每个任务的重复次数，`--concurrency` 是 Harbor trial 并发数。换 harness 使用 `--agent claude-code` 等名称，并配置其兼容的模型服务。自定义 harness 改用 `--agent-import-path my_package.agent:MyAgent`，模块须安装在 Harbor 的 Python 环境中。其他 agent 参数可编辑生成 YAML 的 `kwargs`。凭据和 endpoint 使用对应原生 agent 支持的环境配置，不写入任务或生成 YAML。

更换模型名不会自动转换 API 协议。Codex 与 Claude Code 的模型协议不同，服务必须支持对应的图像、流式返回及工具调用。确需协议适配时，适配器独立版本化并验证，不能静默换模型或丢弃工具调用。此脚本没有注册 `harbor run -d robotworld`，也不代表所有 agent/model 组合都已通过实测。


## ChatGPT 订阅登录：使用 Harbor 原生 Codex

本机 Harbor 0.24.0 已支持读取 Codex 登录凭据。这是 CLI 直接使用订阅登录，不是把订阅包装成 API，也不是领取 OpenAI Platform API 额度。

```bash
codex login
unset OPENAI_API_KEY OPENAI_BASE_URL
export CODEX_FORCE_AUTH_JSON=1
# 非默认登录文件可改用 CODEX_AUTH_JSON_PATH=/私有目录/auth.json。
bash scripts/harbor_run.sh --path tasks/close-drawer --agent codex \
  --model openai/gpt-6-astra --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$PWD/outputs/harbor" --job-name codex-subscription-pilot
```

脚本仅准备资产路径和环境访问凭据，然后原样执行已安装的 `harbor run`。容器准备、内置 agent 执行、产物收集、独立 verifier 和成绩汇总均由 Harbor 负责。Harbor 在 agent 容器内安装 CLI；宿主机 CLI 用于登录。登录凭据不得放进仓库或结果包，模型是否可用以账号权限为准。

全题集使用 `--config configs/jobs/robotworld-builtin-codex.yaml`，先替换其中的模型占位符；该配置固定 CLI 版本。重复次数和并发采用 Harbor 参数，GPU 并发须符合机器容量。`WORLD_CODE_CONTROL` 默认 0，Shell 保持可用。目前支持 `--path` 和任务集 YAML，**尚未注册 `harbor run -d robotworld`**。

验证状态：2026-11-02 已排队启动订阅登录试跑；已确认凭据文件和 Harbor 支持，尚不能声明模型执行验收通过。此入口不修改仿真成功条件，也不把旧自定义循环的成绩视作原生 harness 成绩。


## 环境准备

按主 README 安装 Harbor、仿真镜像及资产。本接入在 Harbor 0.24.0 上开发，其他版本需要验证。在仓库根目录执行：

```bash
export WORLD_ROOT="$PWD"
# 验证资产路径，并在 Docker 以只读方式挂载仓库前创建挂载点。
WORLD_ASSETS="$(python scripts/check_harbor_assets.py "${WORLD_ASSETS:-$PWD/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
export WORLD_NVIDIA_RUNTIME="$(python scripts/harbor_gpu_runtime.py)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export WORLD_CODE_CONTROL=0
unset WORLD_CONTROL_STEP_LIMIT
```

仿真就绪后，任务初始化 agent 容器，在 `/workspace/ROBOTWORLD.md` 写入实时公开说明：目标、机器人、动作参数、单位、参考系和控制器信息。统一 `robotworld` MCP 提供 describe、observe、原生动作、status 和 receipt。

MCP 启动入口从 agent 容器内读取本回合的连接配置，因此不依赖 harness 继承全部环境变量。该文件仅包含公开动作接口的临时凭据，不包含模型密钥或评分器权限，不作为评测产物收集；不要手动上传该文件。

Shell 由所选 harness 提供。`WORLD_CODE_CONTROL=0` 关闭仿真侧反馈程序，不关闭 Shell。

## 直接运行

模型密钥放在环境变量或 Harbor 支持的认证配置里，不要写入 YAML 或任务说明。

```bash
# 外部配置 OPENAI_API_KEY。自定义 OPENAI_BASE_URL 必须容器可达并支持 Responses。
harbor run --path tasks/close-drawer --agent codex \
  --model openai/YOUR_MODEL_ID --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$PWD/outputs/harbor"

# 外部配置 ANTHROPIC_API_KEY；自定义地址须支持 Claude Code 所需协议。
harbor run --path tasks/close-drawer --agent claude-code \
  --model anthropic/YOUR_MODEL_ID --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$PWD/outputs/harbor"
```

全部选题配置：

- `configs/jobs/robotworld-builtin-codex.yaml`：CLI 固定为 0.164.0。
- `configs/jobs/robotworld-builtin-claude-code.yaml`：CLI 固定为 2.1.292。

修改模型和 `n_attempts` 后，运行 `harbor run --config <文件> --jobs-dir "$PWD/outputs/harbor"`。默认单 GPU 任务串行，使用任务既定步数，包括 BEHAVIOR 的既定上限。Harbor 负责安装／准备 CLI，未预装时需要联网。

其他内置 agent 需要支持相应 MCP 传输和图像响应才能使用，不能只凭 Harbor 支持该 agent 就宣称已经兼容。自己的源码 harness 可以实现 Harbor agent 接口，并消费同一 MCP，无需改仿真。

## 评分与可比性

成功、失败、截断和步数上限由仿真环境决定，Harbor 收集最终 episode.json，统一 verifier 映射为分数。文字声称成功不算成功；没有添加新的物理评分条件。

内置 agent 是单独的实验配置：没有旧包装器的自动续聊，也没有旧的请求图像裁剪代理。每次观测包仍有环境声明的历史，但此前观测包是否保留在模型请求中由 harness 决定。未经测量不能宣称提示词、历史或成绩与旧版完全一致。必须记录 harness、CLI、模型和 provider 配置；提前返回记录为 agent_stopped。

## 无模型验收

```bash
harbor run --config configs/jobs/robotworld-native-budget-probe.yaml \
  --jobs-dir "$PWD/outputs/harbor" --job-name native-budget-probe
```

该诊断检查公开观测、受限中性动作、重复请求幂等性、回执、原生终止、归档和评分。它不是解题策略、Oracle 或模型成绩，即使偶然得分也不能计入模型成功率。参考解应另外整理为 solution/solve.sh，不能用直接写分数的脚本替代。

批量使用一个 Harbor job；无人值守时用任务调度器或服务管理器托管，不依赖编辑器终端持续存活。中断及基础设施异常与正常任务失败分开统计。

## 验证状态（2026-10-31）

84 个任务已包含统一启动和 MCP 入口。真实 Harbor 内置 Codex 运行已安装 CLI 0.164.0；清空环境变量后的 MCP 检查成功列出 RoboCasa 的九个工具并调用 describe。模型执行尚未验收：当前接口返回 HTTP 503，独立 Responses 请求也复现了相同错误。Claude Code 已提供标准配置，尚未通过本入口实测。不能把这些证据记为已完成的模型评测。
