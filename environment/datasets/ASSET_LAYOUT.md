# 统一资产布局

`World/Assets/<benchmark>/` 是本轮新增的统一入口。

| 子目录 | 内容 |
|---|---|
| `robodojo/scenes/<task>/Assets/` | 新选 15 道题的布局及本地依赖闭包，保留原始字节 |
| `robodojo/legacy-scenes/` | 旧传送带、硬币资产包，保持历史运行可复现 |
| `behavior_1k/data/` | 官方授权场景子集、机器人、任务实例；仅本地使用，不能再分发 |
| `robocasa/data/` | 官方厨房、物体与机器人数据；保留运行时临时 XML 写权限 |
| `robolab/data/` | 10 道题对应的 9 个 USD 场景及其依赖 |
| `bench2dex/dex2bench_dataset/` | 九道灵巧手任务的实际资产 |
| `bench2dex/anchors/` | 原始场景锚点，不是当前机器人成功示范 |
| `_shared/upstream/dependencies/` | CuRobo、IsaacLab、XPolicyLab 共用依赖中的资源，按来源分类 |
| `<benchmark>/upstream/` | 上游仓库自带且已实体化的资产，硬链接归集，不改上游文件 |

`centralize.py plan` 生成迁移计划，`apply` 原子移动外置资产并建立旧路径相对链接，迁移前后比对完整目录内容散列；`verify` 检查旧路径指向。
`embedded_assets.py --apply` 收集上游自带资产，同文件系统使用硬链接节省存储。清单记录仍未实体化的 Git LFS 指针，不把它们算作已经下载。

资产发布不能直接上传整个 Assets：BEHAVIOR 资产和密钥排除，其他资源按各自来源许可证保留署名和限制。程序不自动上传任何东西。

迁移结束后运行 `python -m environment.datasets.check_mounts`，以无网络、无GPU容器逐项检查旧路径bind mount，并与新目录中抽样文件的SHA256对比。完整目录散列仍由迁移器负责。
