#!/usr/bin/env bash
set -euo pipefail
# robolab selected suite (10 tasks):
# ToolOrganizationTask: 红黑两把锤子放入左侧箱 -> ToolOrganizationTask
# other-tools: 非锤工具放入右侧箱 -> NonHammerToolsInRightBinTask
# cans: 两件罐装食品入箱 -> FoodPacking2CansTask
# cube-left: 魔方放到碗左侧 -> RubiksCubeLeftOfBowlTask
# FruitsOnPlate3Task: 选取恰好三颗水果放到盘子上 -> FruitsOnPlate3Task
# mugs: 选两只杯子放上置物架 -> PutTwoMugsOnShelfTask
# BlockStackingSpecifiedOrderTask: 按红蓝绿黄从下到上堆叠 -> BlockStackingSpecifiedOrderTask
# ClutterPlasticTask: 从杂物中找出全部塑料瓶入箱 -> ClutterPlasticTask
# ReorientWhiteMugsTask: 将所有白色杯子扶正 -> ReorientWhiteMugsTask
# WhiteMugInCenterOfTableTask: 白色杯子放到桌面中央 -> WhiteMugInCenterOfTableTask
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench robolab "$@"
