#!/usr/bin/env bash
set -euo pipefail
# behavior_1k selected suite (10 tasks):
# groceries: 杂货搬运 -> carrying_in_groceries
# desk: 书桌整理 -> clean_up_your_desk
# slicing_vegetables: 切菜 -> slicing_vegetables
# sorting_vegetables: 蔬菜分拣 -> sorting_vegetables
# clean_boxing_gloves: 清洗拳击手套 -> clean_boxing_gloves
# putting_up_Christmas_decorations_inside: 室内圣诞装饰 -> putting_up_Christmas_decorations_inside
# setting_the_table: 双人餐桌布置 -> setting_the_table
# putting_dishes_away_after_cleaning: 餐具集中入柜并关柜门 -> putting_dishes_away_after_cleaning
# can_meat: 肉肠分装封罐 -> can_meat
# freeze_pies: 苹果派分盒冷冻 -> freeze_pies
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench behavior_1k "$@"
