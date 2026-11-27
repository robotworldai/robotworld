# Harbor 接手开发与验收

[English](HARBOR_DEVELOPMENT.md) · [启动入口](../README.zh-CN.md#标准-harbor-入口) · [机器人协议](harness-protocol.md)

## 职责与执行顺序

Harbor 负责 job/trial 调度、agent 安装运行、超时、收集和 verifier 执行；RobotWorld 负责题目、机器人接口和仿真适配；模型服务只负责推理。

`harbor run → main 中的 agent → MCP/HTTP → simulator → 最终 episode.json → collect → 独立 verifier → reward`

agent 只获得公开观测和动作接口。物理状态、成功与失败由 simulator 判断。不要把仿真目录、Docker socket 或评分代码暴露给 agent。Shell 不意味着能直接控制仿真；`WORLD_CODE_CONTROL=1` 只开启规定的反馈控制接口。

## 改一道题

1. 阅读该题 instruction、TOML、compose 和 `tests/task.json`，找到适配器及上游任务，确认 seed、horizon、动作单位/参考系和成功来源。
2. 改对应的权威实现：物理与动作改环境适配层；agent/model 改 job；公开说明改 instruction；收集与墙钟上限改 TOML。不要在 verifier 中另写一份物理判据。
3. 动作语义变化时，同步更新说明和 tool schema：绝对量/增量、单位、参考系、重复执行含义及例子。
4. `environment/harbor_tasks/export.py` 会导出共享客户端、grader 和任务配置。先看 `python scripts/export_harbor_tasks.py --help`，导出到临时目录再审查 diff，避免覆盖手工修改。新增/删题同步维护 catalog 和 job 选题。
5. 任务、运行时或判据变化要记录版本，不静默比较不同定义下的成绩。

## 接新 agent 或 model

Harbor 内置 agent 直接选名称并提供其要求的认证配置，无须修改仿真。自定义 harness 实现固定版本 Harbor 0.24.0 的 Agent 接口，把模块安装到 Harbor 使用的 Python 环境，通过 `--agent-import-path package.module:AgentClass` 选择。

适配器需要在 main 内启动运行时、读取 instruction、连接 MCP（或公开 Episode API）、处理图像/工具回执、传播异常和保留日志。可参考 `agents/robotworld.py` 的生命周期实现，但不要无意继承旧版的续聊或图像裁剪策略；这些属于实验配置。

换模型服务时，先验证文本、图像、流式工具名/参数、工具回执、正常结束。保留请求 ID，禁止静默丢调用、重复不确定动作、擅自换模型，或把协议失败记为任务失败。填写模型名不会自动转换 API 协议。

## 修改后的验收

先运行有针对性的本地测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_make_harbor_job.py tests/test_model_output_replay.py \
  tests/test_harbor_runtime_paths.py tests/test_source_archive.py
```

再按 README 用实际 agent/provider 跑一道题。目录/配置校验和无模型冒烟不能证明模型能行动；需要确认真实机器人工具调用、控制步推进、视频归档，以及独立 verifier 读取了环境结果。

已有有效模型 trial 时，无需再请求模型即可回放：

```bash
# 先按 Harbor 指南设置 WORLD_ROOT / WORLD_ASSETS / WORLD_EPISODE_TOKEN。
python scripts/validate_harbor_model_replay.py \
  --model-config configs/jobs/robotworld-builtin-codex.yaml \
  --task close-drawer --source-trial /absolute/path/to/job/task__trial \
  --output var/validation/my-replay
```

不传 `--source-trial` 时先生成模型轨迹、再回放，此时先替换配置中的模型占位符。`--continue-on-error` 可记录异常后继续其他题，但共享模型服务/配置故障不宜全量重复请求。复用旧 trial 时必须核对场景、种子、运行时和预算；脚本只检查部分元数据，**不能证明初始状态完全相同**。分歧需要排查，不能改分数消除。零动作单独报告；诊断步数截断的结果不能认证完整预算运行。

检查 `pairs.json`、两次 trial 的 `result.json`、`agent/`、`artifacts/simulation/` 和 `verifier/`。配对一致仅证明该次比较一致，不代表任务成功、普遍确定性或物理判据正确。需要成功与失败案例，基础设施异常单列。

## 发布与尚待完善

- 保留任务客户端、grader 副本、Docker 定义、许可证和固定源码校验信息。上游名为 `assets` 的 Python 包属于代码，不能当作场景数据过滤。
- 排除凭据、outputs、缓存和私有参考控制代码；资产按 `docs/ASSETS.md` 及各自许可处理。
- 在发布副本里跑测试、检查 README 链接，中英文同步。
- 当前任务仍通过 WORLD_ROOT 挂载仓库代码；完全不可变的独立镜像和公共 `-d robotworld` 注册尚待完成。
- 当前限制见 `RELEASE_NOTES.md`；没有完整证据前，不宣称 84 题和任意 agent/model 组合均已验收。
