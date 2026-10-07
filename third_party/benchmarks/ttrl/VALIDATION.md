# 验证状态

- 已逐项核对原 TTEnv、eval.py 的 action_joint_names / obs_joint_names、21 维动作、5 帧历史。
- 原 serve 统计跳过两个完整 warmup，锁存 hit/return，在原球重发边界结算；半球不结算。
- 原机器人终止停止，原球重发保留；learned predictor 未实例化。
- 两个入口的 USD 真实闭包为 8 文件，依赖边全部解析；官方环境闭包 9 文件已缓存并锁 SHA256。
- 7 个轻量 CPU 测试通过；额外多批维逆旋转在本机实际Torch2.11 CPU验证通过（无Torch测试venv中该项跳过）。完整固定 checkout 保持 clean。
- Isaac4.5 Docker 镜像已完成构建，日志：`var/runs/docker/native17/builds/ttrl-isaac45-build-retry01.log`；未运行完整 GPU 仿真或 GPT 回合。
- 第一次构建的基础镜像、apt 和 Lab 固定源码获取已完成；pip 因 flatdict 4.0.1 依赖新版 setuptools 已删除的 pkg_resources 失败。外部 Dockerfile 已固定兼容构建后端 setuptools 80.9.0，并提前构建 flatdict；重试日志 `ttrl-isaac45-build-retry01.log`。

## 本机版本证据（2026-09-28）

- nvidia-smi：RTX 5090，driver 595.80，compute capability 12.0。
- 作者 README 明确以 Isaac Sim 4.5.0 开发测试，报告升级 5.0+ 成绩下降。
- World 历史 `var/runs/docker/robolab/isaac51-retry-01/sim50-only/launcher.log` 有 Warp CUDA error 36 和随后崩溃，sim51-only 也记录崩溃。它们是同机旧运行时风险证据，**不等于已经实测 TTRL / Isaac4.5 失败**。
- 本轮保持原4.5镜像独立验证，不静默替换6.0.1；CUDA/渲染实际可用性等待镜像构建后的协调GPU探测。
- 构建完成后的最小 CUDA 实测已确定当前阻塞：`ttrl-isaac45-cuda-kernel.log` 记录原 Torch 2.5.1+cu118 仅支持到 sm_90，RTX5090 sm_120 实际张量内核报 `no kernel image is available for execution on the device`。这是硬件/原运行时阻塞，不是GPT任务失败；未静默升级Torch或Isaac。

## 显式 isaac6 实验档案

- 已另建 `world/ttrl:isaac6.0.1-experimental`，默认4.5镜像与原阻塞证据保留。
- 完整官方Lab2.1.0从原镜像提取到本项目isaaclab21_checkout，固定commit/clean验证通过；不替换成基础镜像的Lab。
- probe01到达原mdp模块导入，新Torch Inductor的CSE泛型声明触发Python3.12 TypeError，尚未任务步进。按该Torch本地eval_frame.py的官方禁用入口设置TORCHDYNAMO_DISABLE=1，使原@torch.compile函数执行原eager实现；不替换奖励或控制公式。该优化执行条件单独记录。
- probe02完成环境构建及初始观测，首步暴露原TTRL aerodynamic adapter向Lab2.1传is_global=True的API不匹配。外部桥保留原力/力矩缓冲，在每次write_data_to_sim传递PhysX原生is_global=True；没有丢弃坐标系标记，也没有只在设置时旋转后固定为局部力。默认局部力分支仍调用原方法。针对连续写入与切回局部力的测试通过，目前共6项CPU测试通过；等待probe03验证。
- probe03已执行真实物理步进，随后原奖励调用缺失的quat_apply_inverse；外部桥补入官方Lab2.3.2同名WXYZ逆旋转公式，附原BSD许可证，奖励调用及公式不改。Torch2.11 CPU实测多维批量旋转方向/shape通过，等待probe04。
- probe04通过：`var/runs/docker/native17/profiles-isaac6/probe04/ttrl/T02`，exit0、infrastructure_ok=true，4个控制步和5个50fps复核视频帧。实际Python3.12.14、IsaacSim6.0.1.0、Torch2.11.0；module_paths确认加载固定Lab2.1源码（基础镜像distribution metadata不代表实际导入源码版本）。首次画面桌面被近景截断，正式模型回合改用项目私有整桌复核相机；仅review镜头，不改变actor、物理或原灯光材质。GPT正式回合尚待运行。

## 实验档正式模型回合

- `var/runs/docker/native17/profiles-isaac6/gpt01/ttrl/T02`：源码Codex app-server，模型gpt-6-astra；原预算540步，实际93步/1.86秒，exit0且infrastructure_ok=true。
- 原非timeout机器人终态停止。原check_reset仅含基座高度<0.50m或x/y位置越界；本轮未记录各真实位置子项，不能指认唯一触发项。最终视频可见身体前倾下沉。后续已在仅evaluator可见的evaluation日志补原阈值各子项，未改actor或评分。
- finished_serves=1、warmup_serves_skipped=1、scored_serves=0，命中率和有效回球率均null；此回合尚未进入正式计分球，不能写0%。whole-episode success仍null。
- apply_action调用3次、coding_control调用6次；连续控制步1..93。完整events与no-images三类条目数逐一相等；逐项核验存于回合`artifact-verification.json`。
- ffprobe确认640×480、50fps、94帧、1.88秒；已人工查看首末图，整桌、球网、机器人和球路均在复核镜头内。录像不送入actor。
- 原4.5镜像与硬件阻塞证据保留；实验6.0.1不宣称官方物理等价，也未加载作者控制策略。
