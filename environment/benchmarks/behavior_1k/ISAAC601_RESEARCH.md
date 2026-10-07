# Isaac Sim 6.0.1 适配调研（2026-09-26）

## 直接可用的上游参考

[PR #2313](https://github.com/StanfordVL/BEHAVIOR-1K/pull/2313) 是上游协作者维护的 isaac-6.0-upgrade 分支，当前 open / 未合并。API 查询 head 为 `850e5ea16cf85d25d4cd26b8910bf33d77fc8fb0`，base 为我们当前 v3.9.3 的 `6cbf70b075816096e9be53958780769f3264d25d`。

虽然标题为 6.0，当前 setup.sh 明确安装 `isaacsim[all,extscache]==6.0.1.0`；simulator 的 KIT_FILES 同时登记 (6,0,0) 和 (6,0,1)。因此这是精确对应 6.0.1 的参考，不只是早期 6.0 猜测。

查询时 PR 最新测试报告为 224 passed / 18 skipped / 0 failed；这是上游报告，不是本机测试，也不代表 carrying_in_groceries 已通过。Profiling 有速度/加载时间回退和部分 N/A，不能宣称完全等价。

[PR #2316](https://github.com/StanfordVL/BEHAVIOR-1K/pull/2316) 已关闭且未合并，针对较早 `6.0.0.1`。作者报告 595 驱动问题被解决、turning_on_radio 的 pi0.5 evaluator 能运行，但指出分割测试失败和可选 teleop 依赖缺失。可作为补充经验，不优先于 #2313。

## 具体适配点

- Python 3.11 → 3.12；Isaac Sim 6.0.1.0 / Kit 110.1；NumPy >=2 与相应编译依赖。不能直接复用官方 3.9.3 镜像内的 Python 3.11 安装。
- 新 `.kit` experience 与 Kit 110 扩展源；显式登记 6.0.1。
- PhysicsContext 私有 PhysX interface 改为 omni.physx 公共 getter。
- USD Sdr 的 GetInput / GetInputNames 更名为 GetShaderInput / GetShaderInputNames。
- timeline STOP 事件回调适配新 dispatcher；不能只假定 event.type 仍存在。
- VisionSensor 在改变分辨率/新增语义后重建 render product、重新连接 annotator，并处理缓冲尚未就绪。RGB-D 与逐帧视频均需要验证。
- 另有资产转换、URDF、lerobot 等改动；对当前评测是否需要，需按调用路径核对。

## World 接入建议

维持 Codex 和现有 v3.9.3 checkout 原样。以 #2313 固定 commit 的差异为依据，在 `environment/benchmarks/behavior_1k/compat/` 实现显式版本检查的外部适配，独立 `.kit` 和 Docker 构建材料放 World；若选择原样使用上游 PR，则另建锁定 commit 的 checkout，而不是覆盖稳定版。不要把 runtime patch 的成功当成已达官方评测条件。

复用已有单场景资产，按空场景 → R1Pro RGB-D/动作 → carrying_in_groceries probe → Codex episode 的顺序验证。保持官方任务实例、判定逻辑、步数预算和事件记录。无需为了调研切换宿主驱动或下载全部资产。

原始 GitHub API 元数据、差异与评论快照：`var/diagnostics/behavior-isaac6-research/pr-2313.json`、`pr-2316.json`。本次只调研，没有应用 PR 或修改上游源码。

[NVIDIA 6.0 迁移入口](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/migration_guides/isaac_sim_6_0/index.html) 提供通用变更说明；具体 OmniGibson 适配以上述上游 PR 为优先参考。
