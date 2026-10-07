# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from robolab.scene_gen.llm_scene_gen.physical_solver import PhysicalSolver
from robolab.scene_gen.llm_scene_gen.predicates import (
    ObjectState,
    PlaceInPredicate,
    PlaceOnPredicate,
    parse_predicates_from_dict,
)


def test_place_on_predicate_parses_shelf_level() -> None:
    pred = parse_predicates_from_dict(
        {
            "type": "place-on",
            "object": "mug",
            "support": "wire_rack",
            "position": "center",
            "shelf_level": 1,
        }
    )

    assert isinstance(pred, PlaceOnPredicate)
    assert pred.shelf_level == 1


def test_place_on_predicate_parses_support_level_alias() -> None:
    pred = parse_predicates_from_dict(
        {
            "type": "place-on",
            "object": "mug",
            "support": "wire_rack",
            "support_level": 2,
        }
    )

    assert isinstance(pred, PlaceOnPredicate)
    assert pred.shelf_level == 2


def test_physical_solver_solves_grouped_place_in_once() -> None:
    place_in = PlaceInPredicate(["lemon_01", "lemon_02"], "bowl")
    object_states = {
        "bowl": ObjectState(name="bowl", x=0.55, y=0.0, z=0.03, yaw=0.0),
        "lemon_01": ObjectState(name="lemon_01", predicates=[place_in]),
        "lemon_02": ObjectState(name="lemon_02", predicates=[place_in]),
    }

    solver = PhysicalSolver()
    success, message = solver.solve(
        object_states=object_states,
        object_dims={
            "bowl": (0.28, 0.28, 0.13),
            "lemon_01": (0.07, 0.05, 0.05),
            "lemon_02": (0.07, 0.05, 0.05),
        },
        object_paths={},
        scene_path="",
    )

    assert success, message
    assert solver.placed_objects == ["lemon_01", "lemon_02"]


def test_place_in_layers_dense_container_contents_inside_bowl_footprint() -> None:
    target_objects = [
        "lemon_01",
        "lemon_02",
        "lemon_03",
        "lemon_04",
        "banana_01",
        "banana_02",
        "banana_03",
        "banana_04",
        "banana_05",
    ]
    place_in = PlaceInPredicate(target_objects, "wooden_bowl")
    object_states = {
        "wooden_bowl": ObjectState(name="wooden_bowl", x=0.43, y=0.22, z=0.065, yaw=30.0),
        **{name: ObjectState(name=name, predicates=[place_in]) for name in target_objects},
    }
    object_dims = {
        "wooden_bowl": (0.280, 0.280, 0.130),
        "lemon_01": (0.076, 0.050, 0.051),
        "lemon_02": (0.076, 0.050, 0.051),
        "lemon_03": (0.060, 0.040, 0.040),
        "lemon_04": (0.060, 0.040, 0.040),
        "banana_01": (0.109, 0.178, 0.037),
        "banana_02": (0.109, 0.178, 0.037),
        "banana_03": (0.109, 0.178, 0.037),
        "banana_04": (0.109, 0.178, 0.037),
        "banana_05": (0.109, 0.178, 0.037),
    }

    solver = PhysicalSolver()
    success, message = solver.solve(
        object_states=object_states,
        object_dims=object_dims,
        object_paths={},
        scene_path="",
    )

    assert success, message
    assert sorted(solver.placed_objects) == sorted(target_objects)

    bowl = object_states["wooden_bowl"]
    yaw = math.radians(bowl.yaw or 0.0)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    bowl_radius = max(object_dims["wooden_bowl"][0], object_dims["wooden_bowl"][1]) / 2
    z_layers = {round(object_states[name].z or 0.0, 2) for name in target_objects}

    assert len(z_layers) >= 2
    for name in target_objects:
        state = object_states[name]
        dx = (state.x or 0.0) - (bowl.x or 0.0)
        dy = (state.y or 0.0) - (bowl.y or 0.0)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        footprint_radius = max(object_dims[name][0], object_dims[name][1]) / 2

        assert math.hypot(local_x, local_y) + footprint_radius <= bowl_radius + 0.01, name


def test_place_on_packs_multiple_centered_objects_on_same_support() -> None:
    fork_pred = PlaceOnPredicate(
        target_object="fork_big",
        support_object="plate_large",
        relative_position="center",
    )
    spoon_pred = PlaceOnPredicate(
        target_object="spoon_big",
        support_object="plate_large",
        relative_position="center",
    )
    object_states = {
        "plate_large": ObjectState(name="plate_large", x=0.5, y=-0.2, z=0.014, yaw=151.0),
        "fork_big": ObjectState(name="fork_big", predicates=[fork_pred]),
        "spoon_big": ObjectState(name="spoon_big", predicates=[spoon_pred]),
    }
    object_dims = {
        "plate_large": (0.331, 0.331, 0.026),
        "fork_big": (0.184, 0.025, 0.013),
        "spoon_big": (0.170, 0.042, 0.017),
    }

    solver = PhysicalSolver()
    success, message = solver.solve(
        object_states=object_states,
        object_dims=object_dims,
        object_paths={},
        scene_path="",
    )

    assert success, message

    plate = object_states["plate_large"]
    support_slots = []
    for name in ("fork_big", "spoon_big"):
        state = object_states[name]
        local_x, local_y = solver._to_local_support_offset(
            plate,
            state.x or 0.0,
            state.y or 0.0,
            plate.yaw or 0.0,
        )
        local_yaw = solver._normalize_yaw((state.yaw or 0.0) - (plate.yaw or 0.0))
        footprint_x, footprint_y = solver._rotated_footprint(object_dims[name], local_yaw)

        assert solver._fits_support_rectangle(
            local_x,
            local_y,
            footprint_x,
            footprint_y,
            object_dims["plate_large"],
        )
        assert not solver._rect_overlaps_layer(
            local_x,
            local_y,
            footprint_x,
            footprint_y,
            support_slots,
            padding=0.004,
        )
        support_slots.append((local_x, local_y, footprint_x, footprint_y))


def test_place_on_shelf_level_uses_shelf_surface_height() -> None:
    solver = PhysicalSolver()
    object_states = {
        "wire_rack": ObjectState(
            name="wire_rack",
            x=0.5,
            y=0.0,
            z=0.25,
            yaw=0.0,
            is_placed=True,
        ),
        "mug": ObjectState(
            name="mug",
            predicates=[
                PlaceOnPredicate(
                    "mug",
                    support_object="wire_rack",
                    relative_position="center",
                    shelf_level=1,
                )
            ],
        ),
    }

    success, message = solver.solve(
        object_states,
        {
            "wire_rack": (0.3, 0.4, 0.5),
            "mug": (0.08, 0.08, 0.1),
        },
        object_paths={},
        scene_path="",
        object_metadata={"wire_rack": {"shelf_levels": [0.2, 0.45]}},
    )

    assert success, message
    assert object_states["mug"].x == 0.5
    assert object_states["mug"].y == 0.0
    assert round(object_states["mug"].z or 0.0, 3) == 0.501


def test_place_on_edge_uses_support_footprint_inset() -> None:
    solver = PhysicalSolver()
    support_dims = (0.241932, 0.4, 0.45412)
    object_dims = (0.086923, 0.106583, 0.079754)
    footprint_inset = 0.01
    object_states = {
        "wire_rack": ObjectState(
            name="wire_rack",
            x=0.397137,
            y=0.25,
            z=0.27076,
            yaw=0.0,
            is_placed=True,
        ),
        "bin": ObjectState(
            name="bin",
            predicates=[
                PlaceOnPredicate(
                    "bin",
                    support_object="wire_rack",
                    relative_position="edge",
                    shelf_level=0,
                )
            ],
        ),
    }

    success, message = solver.solve(
        object_states,
        {
            "wire_rack": support_dims,
            "bin": object_dims,
        },
        object_paths={},
        scene_path="",
        object_metadata={
            "wire_rack": {
                "shelf_levels": [0.2, 0.45],
                "support_footprint_inset_m": footprint_inset,
            }
        },
    )

    assert success, message
    rack = object_states["wire_rack"]
    bin_state = object_states["bin"]
    local_x, local_y = solver._to_local_support_offset(
        rack,
        bin_state.x or 0.0,
        bin_state.y or 0.0,
        rack.yaw or 0.0,
    )
    usable_dims = (
        support_dims[0] - 2 * footprint_inset,
        support_dims[1] - 2 * footprint_inset,
        support_dims[2],
    )

    assert round(local_x, 6) == round((usable_dims[0] - object_dims[0]) / 2, 6)
    assert round(local_y, 6) == 0.0
    assert solver._fits_support_rectangle(
        local_x,
        local_y,
        object_dims[0],
        object_dims[1],
        usable_dims,
    )
