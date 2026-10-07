#!/usr/bin/env bash
set -euo pipefail
# robodojo selected suite (15 tasks):
# hang_mugs: hang_mugs -> hang_mugs
# sweep_blocks: sweep_blocks -> sweep_blocks
# pour_liquid_into_cup: pour_liquid_into_cup -> pour_liquid_into_cup
# make_toast: make_toast -> make_toast
# store_laptop_and_headphones: store_laptop_and_headphones -> store_laptop_and_headphones
# insert_tubes: insert_tubes -> insert_tubes
# plug_in_charger: plug_in_charger -> plug_in_charger
# pour_balls_into_vase: pour_balls_into_vase -> pour_balls_into_vase
# play_Xylophone: play_Xylophone -> play_Xylophone
# fill_pen_holder: fill_pen_holder -> fill_pen_holder
# fill_egg_holder: fill_egg_holder -> fill_egg_holder
# make_kong: make_kong -> make_kong
# pour_by_language: pour_by_language -> pour_by_language
# conveyor: 传送带匹配抓取 -> match_and_pick_from_conveyor
# coin: 硬币投放 -> deposit_coin
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench robodojo "$@"
