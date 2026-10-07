# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical solver for 3D object placement with physics simulation.

This module implements physics-based placement for stacking and containment.
It uses occupancy grids and physics simulation to ensure physically plausible
configurations.
"""

import numpy as np
import random
import math
from typing import Optional, Any
from .predicates import (
    ObjectState,
    PhysicalPredicate,
    PlaceOnPredicate,
    PlaceInPredicate,
    PredicateType,
)


class PhysicalSolver:
    """Solver for physical predicates using physics simulation."""

    def __init__(
        self,
        simulation_app: Optional[Any] = None,
        grid_resolution: float = 0.01,
        stability_threshold: float = 0.02,
    ):
        """Initialize physical solver.

        Args:
            simulation_app: Isaac Sim SimulationApp instance (None for testing)
            grid_resolution: Voxel size for occupancy grid (meters)
            stability_threshold: Maximum displacement for stable placement (meters)
        """
        self.simulation_app = simulation_app
        self.grid_resolution = grid_resolution
        self.stability_threshold = stability_threshold
        self.placed_objects = []  # Track placed objects for occupancy

    def solve(
        self,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
        object_paths: dict[str, str],
        scene_path: str,
        object_metadata: Optional[dict[str, dict[str, Any]]] = None,
    ) -> tuple[bool, str]:
        """Solve physical predicates for all objects.

        Args:
            object_states: Dictionary of object states
            object_dims: Dictionary of object dimensions
            object_paths: Dictionary of USD paths for objects
            scene_path: Path to the base scene

        Returns:
            (success, message) tuple
        """
        if object_metadata is None:
            object_metadata = {}

        # Group predicates by type
        place_on_predicates = []
        place_in_predicates = []
        place_anywhere_predicates = []
        seen_place_in_predicates = set()
        support_placements: dict[tuple[str, Optional[int]], list[tuple[float, float, float, float]]] = {}

        for obj_name, obj_state in object_states.items():
            for pred in obj_state.predicates:
                if isinstance(pred, PlaceOnPredicate):
                    place_on_predicates.append((obj_name, pred))
                elif isinstance(pred, PlaceInPredicate):
                    target_objects = tuple(getattr(pred, "target_objects", [pred.target_object]))
                    key = (pred.support_object, target_objects)
                    if key not in seen_place_in_predicates:
                        seen_place_in_predicates.add(key)
                        place_in_predicates.append((obj_name, pred))
                elif pred.type == PredicateType.PLACE_ANYWHERE:
                    place_anywhere_predicates.append((obj_name, pred))

        # Process place-on predicates by support so sibling objects can be packed together.
        place_on_groups: dict[tuple[str, Optional[int]], list[tuple[str, PlaceOnPredicate]]] = {}
        for obj_name, pred in place_on_predicates:
            place_on_groups.setdefault((pred.support_object, pred.shelf_level), []).append((obj_name, pred))

        for (support_name, shelf_level), group in place_on_groups.items():
            success = self._solve_place_on_group(
                group,
                object_states,
                object_dims,
                support_placements.setdefault((support_name, shelf_level), []),
                object_metadata,
            )
            if not success:
                objects = ", ".join(obj_name for obj_name, _pred in group)
                return False, f"Failed to place {objects} on {support_name}"

        # Process place-in predicates
        for obj_name, pred in place_in_predicates:
            success = self._solve_place_in(pred, object_states, object_dims)
            if not success:
                return False, f"Failed to place objects in {pred.support_object}"

        # Process place-anywhere predicates
        for obj_name, pred in place_anywhere_predicates:
            success = self._solve_place_anywhere(
                object_states[obj_name], object_dims[obj_name], object_states
            )
            if not success:
                return False, f"Failed to place {obj_name} anywhere"

        return True, "All physical constraints resolved"

    def _solve_place_on_group(
        self,
        group: list[tuple[str, PlaceOnPredicate]],
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
        support_slots: list[tuple[float, float, float, float]],
        object_metadata: dict[str, dict[str, Any]],
    ) -> bool:
        """Solve all place-on predicates targeting one support as a joint layout."""
        if not group:
            return True

        support_name = group[0][1].support_object
        support_state = object_states.get(support_name)
        support_dims = object_dims.get(support_name)
        if not support_state or support_state.x is None or support_state.y is None:
            return False
        if support_dims is None:
            return False

        support_metadata = object_metadata.get(support_name, {})
        if len(group) == 1:
            obj_name, pred = group[0]
            return self._solve_place_on(
                object_states[obj_name],
                pred,
                support_state,
                object_dims[obj_name],
                support_dims,
                support_slots,
                support_metadata,
            )

        support_yaw = support_state.yaw or 0.0
        padding = max(self.grid_resolution * 0.5, 0.004)
        occupied_slots = list(support_slots)
        assignment_by_object: dict[str, tuple[float, float, float, float]] = {}
        unset_items = []

        for index, (obj_name, pred) in enumerate(group):
            obj_state = object_states.get(obj_name)
            obj_dims = object_dims.get(obj_name)
            if obj_state is None or obj_dims is None:
                return False

            self._ensure_place_on_orientation(obj_state, pred)
            local_yaw = self._normalize_yaw((obj_state.yaw or 0.0) - support_yaw)
            footprint_x, footprint_y = self._rotated_footprint(obj_dims, local_yaw)
            placement_support_dims = self._placement_support_dims(
                support_dims,
                support_metadata,
            )

            if obj_state.x is not None and obj_state.y is not None:
                local_x, local_y = self._to_local_support_offset(
                    support_state,
                    obj_state.x,
                    obj_state.y,
                    support_yaw,
                )
                if not self._fits_support_rectangle(
                    local_x,
                    local_y,
                    footprint_x,
                    footprint_y,
                    placement_support_dims,
                ):
                    return False
                if self._rect_overlaps_layer(
                    local_x,
                    local_y,
                    footprint_x,
                    footprint_y,
                    occupied_slots,
                    padding,
                ):
                    return False
                occupied_slots.append((local_x, local_y, footprint_x, footprint_y))
                assignment_by_object[obj_name] = (local_x, local_y, footprint_x, footprint_y)
                continue

            candidates = [
                (local_x, local_y)
                for local_x, local_y in self._candidate_support_offsets(
                    pred.relative_position,
                    placement_support_dims,
                    footprint_x,
                    footprint_y,
                )
                if self._fits_support_rectangle(
                    local_x,
                    local_y,
                    footprint_x,
                    footprint_y,
                    placement_support_dims,
                )
            ]
            if not candidates:
                return False
            unset_items.append(
                {
                    "index": index,
                    "name": obj_name,
                    "footprint_x": footprint_x,
                    "footprint_y": footprint_y,
                    "candidates": candidates,
                }
            )

        joint_assignments = self._find_joint_support_slots(unset_items, occupied_slots)
        if joint_assignments is None:
            return False
        assignment_by_object.update(joint_assignments)

        for obj_name, pred in group:
            obj_state = object_states[obj_name]
            obj_dims = object_dims[obj_name]
            local_x, local_y, footprint_x, footprint_y = assignment_by_object[obj_name]
            if obj_state.x is None or obj_state.y is None:
                obj_state.x, obj_state.y = self._to_world_support_offset(
                    support_state,
                    local_x,
                    local_y,
                    support_yaw,
                )
            support_slots.append((local_x, local_y, footprint_x, footprint_y))
            if not self._finish_place_on(
                obj_state,
                pred,
                support_state,
                support_dims,
                obj_dims,
                support_metadata,
            ):
                return False

        return True

    def _find_joint_support_slots(
        self,
        items: list[dict[str, Any]],
        occupied_slots: list[tuple[float, float, float, float]],
    ) -> Optional[dict[str, tuple[float, float, float, float]]]:
        """Backtracking non-overlap assignment for objects sharing one support."""
        padding = max(self.grid_resolution * 0.5, 0.004)
        ordered_items = sorted(
            items,
            key=lambda item: (
                -(item["footprint_x"] * item["footprint_y"]),
                -max(item["footprint_x"], item["footprint_y"]),
                item["index"],
            ),
        )
        assignments: dict[str, tuple[float, float, float, float]] = {}

        def search(
            item_index: int,
            layer_slots: list[tuple[float, float, float, float]],
        ) -> bool:
            if item_index == len(ordered_items):
                return True

            item = ordered_items[item_index]
            footprint_x = item["footprint_x"]
            footprint_y = item["footprint_y"]
            for local_x, local_y in item["candidates"]:
                if self._rect_overlaps_layer(
                    local_x,
                    local_y,
                    footprint_x,
                    footprint_y,
                    layer_slots,
                    padding,
                ):
                    continue

                slot = (local_x, local_y, footprint_x, footprint_y)
                assignments[item["name"]] = slot
                if search(item_index + 1, layer_slots + [slot]):
                    return True
                del assignments[item["name"]]

            return False

        if not search(0, list(occupied_slots)):
            return None
        return dict(assignments)

    def _solve_place_on(
        self,
        obj_state: ObjectState,
        pred: PlaceOnPredicate,
        support_state: Optional[ObjectState],
        obj_dims: tuple[float, float, float],
        support_dims: Optional[tuple[float, float, float]],
        support_slots: Optional[list[tuple[float, float, float, float]]] = None,
        support_metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Solve place-on predicate by stacking object on support."""
        if not support_state or support_state.x is None or support_state.y is None:
            return False
        if support_dims is None:
            return False

        if support_slots is None:
            support_slots = []
        if support_metadata is None:
            support_metadata = {}

        support_yaw = support_state.yaw or 0.0
        placement_support_dims = self._placement_support_dims(
            support_dims,
            support_metadata,
        )

        self._ensure_place_on_orientation(obj_state, pred)

        # Use object's x, y if already set (from spatial predicates)
        if obj_state.x is None or obj_state.y is None:
            local_yaw = self._normalize_yaw((obj_state.yaw or 0.0) - support_yaw)
            footprint_x, footprint_y = self._rotated_footprint(obj_dims, local_yaw)
            slot = self._find_support_slot(
                pred.relative_position,
                placement_support_dims,
                footprint_x,
                footprint_y,
                support_slots,
            )
            if slot is None:
                return False
            obj_state.x, obj_state.y = self._to_world_support_offset(
                support_state,
                slot[0],
                slot[1],
                support_yaw,
            )

        local_x, local_y = self._to_local_support_offset(
            support_state,
            obj_state.x,
            obj_state.y,
            support_yaw,
        )
        local_yaw = self._normalize_yaw((obj_state.yaw or 0.0) - support_yaw)
        footprint_x, footprint_y = self._rotated_footprint(obj_dims, local_yaw)
        if not self._fits_support_rectangle(
            local_x,
            local_y,
            footprint_x,
            footprint_y,
            placement_support_dims,
        ):
            return False
        support_slots.append((local_x, local_y, footprint_x, footprint_y))

        return self._finish_place_on(
            obj_state,
            pred,
            support_state,
            support_dims,
            obj_dims,
            support_metadata,
        )

    def _ensure_place_on_orientation(
        self,
        obj_state: ObjectState,
        pred: PlaceOnPredicate,
    ) -> None:
        # Set orientation before selecting an occupancy slot so the footprint is accurate.
        if obj_state.yaw is None:
            if pred.stability_preference == "stable":
                obj_state.yaw = 0.0
            elif pred.stability_preference == "unstable":
                obj_state.yaw = random.uniform(0, 360)
            else:
                obj_state.yaw = random.choice([0, 90, 180, 270])

    def _finish_place_on(
        self,
        obj_state: ObjectState,
        pred: PlaceOnPredicate,
        support_state: ObjectState,
        support_dims: tuple[float, float, float],
        obj_dims: tuple[float, float, float],
        support_metadata: dict[str, Any],
    ) -> bool:
        support_z_top = self._support_surface_z(support_state, support_dims, pred, support_metadata)
        if support_z_top is None:
            return False

        obj_state.z = support_z_top + obj_dims[2] / 2 + 0.001  # Small gap

        if obj_state.pitch is None:
            obj_state.pitch = 0.0
        if obj_state.roll is None:
            obj_state.roll = 0.0

        obj_state.is_placed = True
        self.placed_objects.append(obj_state.name)
        return True

    def _support_surface_z(
        self,
        support_state: ObjectState,
        support_dims: tuple[float, float, float],
        pred: PlaceOnPredicate,
        support_metadata: dict[str, Any],
    ) -> Optional[float]:
        if pred.shelf_level is not None:
            levels = support_metadata.get("shelf_levels") or []
            try:
                level_index = int(pred.shelf_level)
            except (TypeError, ValueError):
                return None
            if level_index < 0 or level_index >= len(levels):
                return None
            if support_state.z is None:
                support_bottom = 0.0
            else:
                support_bottom = support_state.z - support_dims[2] / 2
            return support_bottom + float(levels[level_index])

        # Set z height on top of support
        if support_state.z is None:
            # Assume support is on table at z=0
            return support_dims[2] if support_dims else 0.05
        return support_state.z + (support_dims[2] / 2 if support_dims else 0.05)

    def _placement_support_dims(
        self,
        support_dims: tuple[float, float, float],
        support_metadata: dict[str, Any],
    ) -> tuple[float, float, float]:
        inset = self._support_footprint_inset(support_metadata)
        if inset <= 0.0:
            return support_dims
        return (
            max(0.0, support_dims[0] - 2 * inset),
            max(0.0, support_dims[1] - 2 * inset),
            support_dims[2],
        )

    @staticmethod
    def _support_footprint_inset(support_metadata: dict[str, Any]) -> float:
        try:
            inset = float(support_metadata.get("support_footprint_inset_m", 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(inset):
            return 0.0
        return max(0.0, inset)

    def _find_support_slot(
        self,
        relative_position: Optional[str],
        support_dims: tuple[float, float, float],
        footprint_x: float,
        footprint_y: float,
        support_slots: list[tuple[float, float, float, float]],
    ) -> Optional[tuple[float, float]]:
        padding = max(self.grid_resolution * 0.5, 0.004)
        for local_x, local_y in self._candidate_support_offsets(
            relative_position,
            support_dims,
            footprint_x,
            footprint_y,
        ):
            if not self._fits_support_rectangle(
                local_x,
                local_y,
                footprint_x,
                footprint_y,
                support_dims,
            ):
                continue
            if self._rect_overlaps_layer(
                local_x,
                local_y,
                footprint_x,
                footprint_y,
                support_slots,
                padding,
            ):
                continue
            return local_x, local_y
        return None

    def _candidate_support_offsets(
        self,
        relative_position: Optional[str],
        support_dims: tuple[float, float, float],
        footprint_x: float,
        footprint_y: float,
    ) -> list[tuple[float, float]]:
        half_x = max(0.0, support_dims[0] / 2 - footprint_x / 2)
        half_y = max(0.0, support_dims[1] / 2 - footprint_y / 2)
        offsets: list[tuple[float, float]] = []

        if relative_position == "edge":
            offsets.extend(
                [
                    (half_x, 0.0),
                    (-half_x, 0.0),
                    (0.0, half_y),
                    (0.0, -half_y),
                    (half_x, half_y),
                    (half_x, -half_y),
                    (-half_x, half_y),
                    (-half_x, -half_y),
                ]
            )

        offsets.append((0.0, 0.0))
        for fraction, count in [(0.35, 8), (0.55, 12), (0.75, 16), (1.0, 20)]:
            for index in range(count):
                angle = 2 * math.pi * index / count
                offsets.append(
                    (
                        half_x * fraction * math.cos(angle),
                        half_y * fraction * math.sin(angle),
                    )
                )

        unique = []
        seen = set()
        for local_x, local_y in offsets:
            key = (round(local_x, 6), round(local_y, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append((local_x, local_y))
        return unique

    def _fits_support_rectangle(
        self,
        local_x: float,
        local_y: float,
        footprint_x: float,
        footprint_y: float,
        support_dims: tuple[float, float, float],
    ) -> bool:
        return (
            abs(local_x) + footprint_x / 2 <= support_dims[0] / 2 + 1e-9
            and abs(local_y) + footprint_y / 2 <= support_dims[1] / 2 + 1e-9
        )

    def _to_local_support_offset(
        self,
        support_state: ObjectState,
        world_x: float,
        world_y: float,
        support_yaw: float,
    ) -> tuple[float, float]:
        dx = world_x - (support_state.x or 0.0)
        dy = world_y - (support_state.y or 0.0)
        yaw = math.radians(support_yaw)
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return local_x, local_y

    def _to_world_support_offset(
        self,
        support_state: ObjectState,
        local_x: float,
        local_y: float,
        support_yaw: float,
    ) -> tuple[float, float]:
        yaw = math.radians(support_yaw)
        world_x = (support_state.x or 0.0) + local_x * math.cos(yaw) - local_y * math.sin(yaw)
        world_y = (support_state.y or 0.0) + local_x * math.sin(yaw) + local_y * math.cos(yaw)
        return world_x, world_y

    def _solve_place_in(
        self,
        pred: PlaceInPredicate,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
    ) -> bool:
        """Solve place-in by packing objects above a container mouth.

        Dense containers should overflow upward in layers, not sideways through
        the container wall. Physics settling can then drop the layered pile into
        the bowl/bin without starting from an invalid horizontal spread.
        """
        container_state = object_states.get(pred.support_object)
        if not container_state or container_state.x is None or container_state.y is None:
            return False

        container_dims = object_dims.get(pred.support_object)
        if not container_dims:
            return False

        target_objects = list(getattr(pred, "target_objects", [pred.target_object]))
        pack_items = []
        for index, obj_name in enumerate(target_objects):
            obj_state = object_states.get(obj_name)
            if not obj_state:
                continue
            obj_dims = object_dims.get(obj_name)
            if not obj_dims:
                return False
            pack_items.append((obj_name, obj_state, obj_dims, index))

        if not pack_items:
            return False

        container_yaw = container_state.yaw or 0.0
        mouth_radius_x = max(container_dims[0] * 0.43, self.grid_resolution)
        mouth_radius_y = max(container_dims[1] * 0.43, self.grid_resolution)
        container_top_z = self._container_top_z(container_state, container_dims)
        layer_gap = max(self.grid_resolution * 2.5, 0.025)
        overlap_padding = max(self.grid_resolution * 0.5, 0.004)

        pack_items.sort(
            key=lambda item: (
                -(item[2][0] * item[2][1]),
                -max(item[2][0], item[2][1]),
                item[3],
            )
        )

        layers: list[dict[str, Any]] = []

        for obj_name, obj_state, obj_dims, _index in pack_items:
            local_yaws = self._candidate_local_yaws(obj_state.yaw, container_yaw)
            slot = None
            layer_index = None

            for index, layer in enumerate(layers):
                slot = self._find_container_slot(
                    obj_dims,
                    local_yaws,
                    mouth_radius_x,
                    mouth_radius_y,
                    layer["rects"],
                    index,
                    overlap_padding,
                )
                if slot is not None:
                    layer_index = index
                    break

            if slot is None:
                layer_index = len(layers)
                bottom_z = (
                    container_top_z + 0.02
                    if not layers
                    else layers[-1]["bottom_z"] + layers[-1]["height"] + layer_gap
                )
                layers.append({"bottom_z": bottom_z, "height": 0.0, "rects": []})
                slot = self._find_container_slot(
                    obj_dims,
                    local_yaws,
                    mouth_radius_x,
                    mouth_radius_y,
                    layers[layer_index]["rects"],
                    layer_index,
                    overlap_padding,
                )

            if slot is None:
                local_yaw, footprint_x, footprint_y = self._best_container_yaw(
                    obj_dims,
                    local_yaws,
                    mouth_radius_x,
                    mouth_radius_y,
                )
                local_x = 0.0
                local_y = 0.0
            else:
                local_x, local_y, local_yaw, footprint_x, footprint_y = slot

            layer = layers[layer_index]
            layer["rects"].append((local_x, local_y, footprint_x, footprint_y))
            layer["height"] = max(layer["height"], obj_dims[2])

            obj_state.x, obj_state.y = self._to_world_container_offset(
                container_state,
                local_x,
                local_y,
                container_yaw,
            )
            obj_state.z = layer["bottom_z"] + obj_dims[2] / 2

            obj_state.yaw = self._normalize_yaw(container_yaw + local_yaw)
            if obj_state.pitch is None:
                obj_state.pitch = 0.0
            if obj_state.roll is None:
                obj_state.roll = 0.0

            obj_state.is_placed = True
            self.placed_objects.append(obj_name)

        return True

    def _container_top_z(
        self,
        container_state: ObjectState,
        container_dims: tuple[float, float, float],
    ) -> float:
        container_center_z = (
            container_state.z if container_state.z is not None else container_dims[2] / 2
        )
        return container_center_z + container_dims[2] / 2

    def _find_container_slot(
        self,
        obj_dims: tuple[float, float, float],
        local_yaws: list[float],
        radius_x: float,
        radius_y: float,
        layer_rects: list[tuple[float, float, float, float]],
        layer_index: int,
        padding: float,
    ) -> Optional[tuple[float, float, float, float, float]]:
        for local_x, local_y in self._candidate_container_offsets(radius_x, radius_y, layer_index):
            for local_yaw in local_yaws:
                footprint_x, footprint_y = self._rotated_footprint(obj_dims, local_yaw)
                if not self._fits_container_ellipse(
                    local_x,
                    local_y,
                    footprint_x,
                    footprint_y,
                    radius_x,
                    radius_y,
                ):
                    continue
                if self._rect_overlaps_layer(
                    local_x,
                    local_y,
                    footprint_x,
                    footprint_y,
                    layer_rects,
                    padding,
                ):
                    continue
                return (local_x, local_y, local_yaw, footprint_x, footprint_y)
        return None

    def _candidate_local_yaws(
        self,
        world_yaw: Optional[float],
        container_yaw: float,
    ) -> list[float]:
        candidates = []
        if world_yaw is not None:
            candidates.append(self._normalize_yaw(world_yaw - container_yaw))
        candidates.extend(
            [
                0.0,
                30.0,
                45.0,
                60.0,
                90.0,
                120.0,
                135.0,
                150.0,
                180.0,
                210.0,
                225.0,
                240.0,
                270.0,
                300.0,
                315.0,
                330.0,
            ]
        )

        unique = []
        seen = set()
        for yaw in candidates:
            normalized = self._normalize_yaw(yaw)
            key = round(normalized, 6)
            if key not in seen:
                seen.add(key)
                unique.append(normalized)
        return unique

    def _candidate_container_offsets(
        self,
        radius_x: float,
        radius_y: float,
        layer_index: int,
    ) -> list[tuple[float, float]]:
        offsets = []
        if layer_index > 0:
            stagger_radius = min(radius_x, radius_y) * min(0.12 + 0.04 * (layer_index % 3), 0.22)
            stagger_angle = math.radians((layer_index * 137.5) % 360)
            offsets.append(
                (
                    stagger_radius * math.cos(stagger_angle),
                    stagger_radius * math.sin(stagger_angle),
                )
            )

        offsets.append((0.0, 0.0))

        for radius_fraction, count in [(0.22, 8), (0.38, 10), (0.54, 12), (0.68, 16)]:
            for index in range(count):
                angle = 2 * math.pi * index / count + math.radians(layer_index * 23.0)
                offsets.append(
                    (
                        radius_x * radius_fraction * math.cos(angle),
                        radius_y * radius_fraction * math.sin(angle),
                    )
                )

        return offsets

    def _rotated_footprint(
        self,
        obj_dims: tuple[float, float, float],
        yaw_degrees: float,
    ) -> tuple[float, float]:
        yaw = math.radians(yaw_degrees)
        cos_yaw = abs(math.cos(yaw))
        sin_yaw = abs(math.sin(yaw))
        footprint_x = obj_dims[0] * cos_yaw + obj_dims[1] * sin_yaw
        footprint_y = obj_dims[0] * sin_yaw + obj_dims[1] * cos_yaw
        return footprint_x, footprint_y

    def _fits_container_ellipse(
        self,
        local_x: float,
        local_y: float,
        footprint_x: float,
        footprint_y: float,
        radius_x: float,
        radius_y: float,
    ) -> bool:
        normalized_x = (abs(local_x) + footprint_x / 2) / radius_x
        normalized_y = (abs(local_y) + footprint_y / 2) / radius_y
        return normalized_x**2 + normalized_y**2 <= 1.0

    def _rect_overlaps_layer(
        self,
        local_x: float,
        local_y: float,
        footprint_x: float,
        footprint_y: float,
        layer_rects: list[tuple[float, float, float, float]],
        padding: float,
    ) -> bool:
        for other_x, other_y, other_footprint_x, other_footprint_y in layer_rects:
            overlap_x = abs(local_x - other_x) < (footprint_x + other_footprint_x) / 2 + padding
            overlap_y = abs(local_y - other_y) < (footprint_y + other_footprint_y) / 2 + padding
            if overlap_x and overlap_y:
                return True
        return False

    def _best_container_yaw(
        self,
        obj_dims: tuple[float, float, float],
        local_yaws: list[float],
        radius_x: float,
        radius_y: float,
    ) -> tuple[float, float, float]:
        best_yaw = local_yaws[0]
        best_footprint = self._rotated_footprint(obj_dims, best_yaw)
        best_overflow = float("inf")

        for local_yaw in local_yaws:
            footprint_x, footprint_y = self._rotated_footprint(obj_dims, local_yaw)
            overflow = max(0.0, footprint_x / 2 - radius_x) + max(
                0.0,
                footprint_y / 2 - radius_y,
            )
            if overflow < best_overflow:
                best_yaw = local_yaw
                best_footprint = (footprint_x, footprint_y)
                best_overflow = overflow

        return best_yaw, best_footprint[0], best_footprint[1]

    def _to_world_container_offset(
        self,
        container_state: ObjectState,
        local_x: float,
        local_y: float,
        container_yaw: float,
    ) -> tuple[float, float]:
        yaw = math.radians(container_yaw)
        world_x = (container_state.x or 0.0) + local_x * math.cos(yaw) - local_y * math.sin(yaw)
        world_y = (container_state.y or 0.0) + local_x * math.sin(yaw) + local_y * math.cos(yaw)
        return world_x, world_y

    def _normalize_yaw(self, yaw_degrees: float) -> float:
        return yaw_degrees % 360.0

    def _solve_place_anywhere(
        self,
        obj_state: ObjectState,
        obj_dims: tuple[float, float, float],
        all_states: dict[str, ObjectState],
    ) -> bool:
        """Solve place-anywhere by finding random supported position."""
        # For simplicity, place on table if no other placement found
        # In full implementation, this would check all possible support surfaces

        if obj_state.x is None or obj_state.y is None:
            # Find a random non-colliding position
            for _ in range(20):
                obj_state.x = random.uniform(-0.3, 0.3)
                obj_state.y = random.uniform(-0.3, 0.3)

                # Check if collides with already placed objects
                collision = False
                for other_name, other_state in all_states.items():
                    if other_name == obj_state.name or not other_state.is_placed:
                        continue
                    if other_state.x is None or other_state.y is None:
                        continue

                    dist = np.sqrt(
                        (obj_state.x - other_state.x) ** 2
                        + (obj_state.y - other_state.y) ** 2
                    )
                    if dist < 0.1:
                        collision = True
                        break

                if not collision:
                    break

        # Place on table surface
        if obj_state.z is None:
            obj_state.z = obj_dims[2] / 2 + 0.001

        if obj_state.yaw is None:
            obj_state.yaw = random.uniform(0, 360)
        if obj_state.pitch is None:
            obj_state.pitch = 0.0
        if obj_state.roll is None:
            obj_state.roll = 0.0

        obj_state.is_placed = True
        self.placed_objects.append(obj_state.name)

        return True

    def validate_with_physics(
        self, scene_path: str, num_steps: int = 300
    ) -> tuple[bool, dict]:
        """Validate scene stability using physics simulation.

        This method uses Isaac Sim to run a physics simulation and check if
        objects remain stable or fall/move significantly.

        Args:
            scene_path: Path to the USD scene file
            num_steps: Number of simulation steps to run (~5s at 60Hz)

        Returns:
            (is_stable, diagnostics) tuple
        """
        if not self.simulation_app:
            # No simulation available, assume stable
            return True, {"message": "No physics validation (simulation not available)"}

        try:
            import omni.usd
            import omni.timeline
            from pxr import UsdGeom, Gf

            # Load scene
            print(
                f"[PhysicalSolver] Loading scene for physics validation: {scene_path}"
            )
            omni.usd.get_context().open_stage(scene_path)
            stage = omni.usd.get_context().get_stage()

            # Record initial positions of all objects (excluding tables and static objects)
            initial_positions = {}
            for prim in stage.Traverse():
                prim_name = prim.GetName().lower()

                # Skip tables, ground plane, and scene itself
                if any(
                    skip in prim_name
                    for skip in ["table", "ground", "scene", "physics", "render"]
                ):
                    continue

                # Only track Xforms that are likely to be objects
                if prim.IsA(UsdGeom.Xform):
                    xformable = UsdGeom.Xformable(prim)
                    xform_ops = xformable.GetOrderedXformOps()

                    # Find translate op
                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            initial_pos = op.Get()
                            if initial_pos:
                                initial_positions[prim.GetPath()] = initial_pos
                            break

            if not initial_positions:
                return True, {"message": "No objects found to validate"}

            print(f"[PhysicalSolver] Tracking {len(initial_positions)} objects")

            # Run physics simulation (similar to settle_scenes.py)
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            for _ in range(num_steps):
                self.simulation_app.update()
            timeline.pause()

            print(f"[PhysicalSolver] Simulation complete, checking stability...")

            # Check final positions
            final_positions = {}
            for prim_path in initial_positions:
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    xformable = UsdGeom.Xformable(prim)
                    xform_ops = xformable.GetOrderedXformOps()

                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            final_pos = op.Get()
                            if final_pos:
                                final_positions[prim_path] = final_pos
                            break

            # Calculate displacements
            unstable_objects = []
            max_displacement = 0.0

            for prim_path, initial_pos in initial_positions.items():
                final_pos = final_positions.get(prim_path)
                if final_pos:
                    displacement = np.linalg.norm(
                        np.array(final_pos) - np.array(initial_pos)
                    )
                    max_displacement = max(max_displacement, displacement)

                    if displacement > self.stability_threshold:
                        obj_name = str(prim_path).split("/")[-1]
                        unstable_objects.append(
                            {
                                "object": obj_name,
                                "displacement": float(displacement),
                                "initial": [float(x) for x in initial_pos],
                                "final": [float(x) for x in final_pos],
                            }
                        )
                        print(
                            f"[PhysicalSolver]   {obj_name}: moved {displacement:.4f}m"
                        )

            is_stable = len(unstable_objects) == 0

            if is_stable:
                print(
                    f"[PhysicalSolver] ✓ Scene is stable (max displacement: {max_displacement:.4f}m)"
                )
            else:
                print(
                    f"[PhysicalSolver] ✗ Scene is unstable ({len(unstable_objects)} objects moved)"
                )

            diagnostics = {
                "stable": is_stable,
                "num_objects": len(initial_positions),
                "unstable_objects": unstable_objects,
                "max_displacement": float(max_displacement),
            }

            return is_stable, diagnostics

        except Exception as e:
            import traceback

            print(f"[PhysicalSolver] Error during physics validation: {e}")
            traceback.print_exc()
            return False, {"error": str(e)}

    def settle_scene(self, scene_path: str, output_path: str, num_steps: int = 300):
        """Settle a scene using physics simulation and save the result.

        This is similar to the settle_scenes.py utility but integrated into
        the scene generation pipeline.

        Args:
            scene_path: Path to input USD scene
            output_path: Path to save settled scene
            num_steps: Number of simulation steps
        """
        if not self.simulation_app:
            print("[PhysicalSolver] No simulation app, cannot settle scene")
            return

        try:
            import omni.usd
            import omni.timeline

            print(f"[PhysicalSolver] Settling scene: {scene_path}")

            # Open scene
            omni.usd.get_context().open_stage(scene_path)
            timeline = omni.timeline.get_timeline_interface()

            # Run physics
            timeline.play()
            for _ in range(num_steps):
                self.simulation_app.update()
            timeline.pause()

            # Export settled scene
            stage = omni.usd.get_context().get_stage()
            stage.GetRootLayer().Export(output_path)

            print(f"[PhysicalSolver] Settled scene saved to: {output_path}")

        except Exception as e:
            print(f"[PhysicalSolver] Error settling scene: {e}")
            import traceback

            traceback.print_exc()
