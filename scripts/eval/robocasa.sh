#!/usr/bin/env bash
set -euo pipefail
# robocasa selected suite (10 tasks):
# 8: 台面物品分类归位 -> CountertopCleanup
# 9: 杯碗分流归位并关柜 -> SortingCleanup
# 10: 咖啡杯对准出液口 -> CoffeeSetupMug
# 14: 关闭指定抽屉 -> CloseDrawer
# 17: 导航到指定厨房设施 -> NavigateKitchen
# PackIdenticalLunches: 按数量分装两份相同午餐 -> PackIdenticalLunches
# OrganizeMugsByHandle: 杯柄朝右放入柜中 -> OrganizeMugsByHandle
# LoadDishwasher: 装载洗碗机并关门 -> LoadDishwasher
# MicrowaveCorrectMeal: 按食物选碗并启动微波炉 -> MicrowaveCorrectMeal
# ResetCabinetDoors: 关闭所有开着的柜门 -> ResetCabinetDoors
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench robocasa "$@"
