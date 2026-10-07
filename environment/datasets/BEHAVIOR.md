# BEHAVIOR 官方数据

数据由上游提供：`behavior-1k/zipped-datasets`，当前源码版本要求 `behavior-1k-assets-3.9.0.zip`、`omnigibson-robot-assets-3.8.2.zip`、`2026-challenge-task-instances.zip`。按官方工具下载和解压，保留上游目录结构；不从 RoboDojo 复制资产。

目标根目录：`World/var/datasets/behavior_1k`；需包含 `behavior-1k-assets/`、`omnigibson-robot-assets/`、`2026-challenge-task-instances/` 与官方工具生成的 `omnigibson.key`。

单场景下载：`python -m environment.datasets.behavior_1k.prepare_scene` 按 ZIP 区段提取目标任务依赖，不下载完整归档。仅准备 carrying_in_groceries、R1Pro、public index 10（实例 311）。

准备入口：`environment/containers/behavior_1k/prepare-assets.sh`。必须先接受 BEHAVIOR 数据许可，再显式传入 --accept-license。本次用户已于 2026-09-26 接受，仅用于非商业学术研究。许可和场景资产独立于 MIT 源码，禁止把这套数据和 key 打包到之前的 upload-hf 或 GitHub。详见上游 `OmniGibson/omnigibson/utils/asset_utils.py::print_user_agreement`。

初次准备约 616 MB 的单场景、R1Pro 和任务文件，以及官方工具生成的密钥；下载清单保存在数据根目录，记录来源 revision、大小、CRC/SHA256。未下载完整 31 GB 场景归档。后续真实加载补齐了下述依赖，Isaac 6.0.1 下已完成实例 probe 和短回合。若新动作触发未准备的资源，继续仅补充实际依赖。

Isaac 6 probe 后补充：当前官方房间清单包含 corridor_0、garage_0、garden_0、kitchen_0、living_room_0，共选定 139 个物体模型。旧 partial_rooms JSON 不足以作为完整依赖清单。另需场景粒子模型与官方全局系统定义；后者仅补 metadata 和模型路径索引，未下载无关模型本体。详见下载器 systems.py 与数据目录 README。
