# 9 项需求与官方任务的对应范围

以下名称是 RobotWorld 任务描述，不能直接当作 RoboLab 的官方 task ID。候选类已核对存在；“存在”不等于已完成 GPU 验证。当前只验证了 ToolOrganizationBothTask 的工具闭环，未完成整回合成功率评测。

| ID | 需求 | 相关官方类 | 当前边界 |
|---|---|---|---|
| 11 | 工具按类别和数量放入多个容器 | `ToolOrganizationBothTask`, `NonHammerToolsInRightBinTask`, `FoodPacking2CansTask` | 相关分类/数量任务，尚无精确组合映射 |
| 12 | 根据左/右、前/后和数量完成桌面重排 | `RubiksCubeLeftOfBowlTask`, `RubiksCubeInFrontOfBowlTask`, `PutTwoMugsOnShelfTask` | 对应不同空间关系/数量任务，不是一个官方组合任务 |
| 13 | 容器已满时选择备用容器 |  | 尚未找到精确官方映射 |
| 16 | 多物体搬运中途发生掉落后的续作 | `ToolOrganizationBothTask` | 可记录自然掉落后的续作；没有注入掉落 |
| 29 | 抓取滑落后的重新定位与恢复 | `ToolOrganizationBothTask` | 可记录自然滑落后的恢复；没有注入滑落 |
| 31 | 放置偏离目标区域后的纠正 | `ToolOrganizationBothTask` | 可记录自然放置误差后的纠正；没有注入错误放置 |
| 32 | 命令 completed 但物体没有达到目标状态 | `ToolOrganizationBothTask` | 工具执行完成与任务成功分开；没有注入虚假成功反馈 |
| 33 | 短暂传感器缺失后的安全继续 |  | 需要独立传感器缺失协议；原始观测不改 |
| 36 | 主动终态检查后发现未完成并补做 | `ToolOrganizationBothTask` | 可观察模型的终态验证与补做；没有改变官方判据 |

原始配置、机器人和判据保持上游版本。需要固定的满容器条件、强制掉落、传感器中断等挑战时，应另建 World 自定义协议，单独命名和计分，不混入官方结果。机器可读版本为 robotworld-tasks.json。
