# Isaac 6.0.1 原渲染路径验证（2026-09-26）

## 已验证

保留官方 `RealTimePathTracing`、原场景资产、灯光与曝光，未使用替代渲染器。上游源码保持干净，Isaac API 差异仍通过外部兼容层适配；这不代表原版 Isaac 5.1 比赛镜像已通过。

运行记录均位于 `World/var/runs/docker/behavior_1k/`：

| 运行 | 验证范围 | 结果 |
|---|---|---|
| groceries-render-baseline10 | 原渲染器、既有 IK 配置、实例311重置 | exit 0 |
| groceries-render-native11 | 原渲染器、官方 R1Pro JointController（23维）、重置 | exit 0，37次显式 render 全部返回 |
| groceries-render-native12 | 同上，再连续渲染60帧，不推进物理 | exit 0，三路独立视频，各60帧、30fps |
| groceries-render-light-audit14 | 官方机器人、原渲染器，灯光位置/继承可见性审计与60帧复测 | exit 0 |
| groceries-render-codex13 | 原渲染器、外部 IK 工具、本地源码 Codex | exit 0，181步、7次工具请求（一次越界请求被拒绝），任务 success=false/Q=0 |

native12 的连续渲染耗时 17.6–35.2ms/帧。初始化中最长一次显式 render 约12.1秒，初始化/重置总计约2分40秒；此前首帧卡住未在这几次复测中重现。不能把当前通过归因于一个未经验证的“死锁修复”：没有修改渲染算法，缓存/启动状态的具体影响尚未建立因果对照。

codex13 的官方视频为30fps、181帧；完整 events 与 no-images 序号/类型匹配：codex 89、tools 14、environment 544，其中181条 action_requested 与181条 action_completed。该回合仅测试连通与渲染，未完成杂货任务，也未使用完整官方步数预算。

## 暗光状态

原渲染器的原始 PNG 仍偏暗，不能称作已修复暗光。实际设置：tonemap op=6，ISO=100，f-number=5，shutter=50，自动曝光关闭。天空纹理已加载，车库顶棚 `ceilings_isizwh_0` 内嵌4个光源，各 intensity=10000；不能因场景清单没有独立车库灯具而断言没有光源。

native12 的头部图像 RGB 均值从首帧29.12变化到第60帧35.78（0–255），说明输出仍有时序收敛；这不是曝光或图像后处理修改。尚无同实例5.1正常运行的参考图，不能断言偏暗全是原环境本身，也不能断言全是6.0.1缺陷。audit14 再次通过：车库4个顶棚光源有效可见，USD 与仿真中的位置差异小于0.00002m，未发现光源隐藏或位置同步错误。该证据排除了这两种候选原因，尚不能证明与5.1画面完全等价。继续以原灯光/曝光为基准，不加灯、不删除遮挡、不提高ISO、不对模型输入提亮。

原始证据：`groceries-render-native12/render-state.json`、`render-events.jsonl`、`render-probe/summary.json`，以及 `render-probe/{head,left_wrist,right_wrist}.mp4` 和首尾 PNG。这三段视频是静止渲染探测，不是任务动作视频；官方任务视频仍采用左侧上下腕部、右侧头部的拼接布局。

## 入口约束

`--renderer` 现在只接受 `upstream`，外部兼容层也拒绝旧替代值；RayTracedLighting 替换分支已删除。历史实验数据不删除。

`--probe-only` 默认读取原版 `OmniGibson/omnigibson/eval/r1pro.yaml`，不再悄悄使用 IK 控制器。模型运行仍使用 World 的 IK 配置，工具语义尚未改为原版23维关节控制。

`--render-diagnostics` 记录显式 render 的开始/结束及耗时；该记录不覆盖引擎内部所有 render 路径。probe 另外保存实际渲染设置、光源/相机属性及60帧独立视频，物理步数为0。

## 复现

从 World 运行，output 必须使用新目录：

```bash
python3 environment/containers/behavior_1k/run.py \
  --image world/behavior:isaac6.0.1-experimental --isaac601-compat \
  --render-diagnostics --probe-only \
  --assets "$PWD/var/datasets/behavior_1k" \
  --codex-home "$PWD/var/auth/robodojo-codex" \
  --output "$PWD/var/runs/docker/behavior_1k/new-render-probe"
```

模型短回合去掉 `--probe-only`，加 `--max-steps 180 --max-actions 12 --timeout 600`。使用自己的 Codex 登录目录。默认不设置 max-steps 时沿用官方任务预算。
