# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-stack version checks.

Recordings carry ``isaaclab_version`` / ``isaacsim_version`` / ``recorded_at``
HDF5 attrs stamped at record time (see
``robolab.core.logging.streaming_hdf5_handler``), loaded via
``robolab.core.utils.file_utils.load_hdf5_provenance``. These checks compare
them against the currently installed stack.
"""

import importlib.metadata

from robolab.core.utils.file_utils import load_hdf5_provenance


def warn_on_stack_mismatch(hdf5_path: str) -> None:
    """Print a notice when a recording was made on a different IsaacSim/IsaacLab stack.

    Recorded outcomes are not invariant across simulator versions: different
    IsaacSim/IsaacLab releases ship different PhysX builds, so contact dynamics
    evolve between versions and an open-loop replay may play out differently
    than the original episode even from an identical initial state.
    """
    recorded = load_hdf5_provenance(hdf5_path)
    if not recorded.get("isaaclab_version") and not recorded.get("isaacsim_version"):
        print("\033[93mNOTE: recording has no simulator-version provenance (recorded before "
              "version stamping). Recorded outcomes are not invariant across simulator "
              "versions; if this file was recorded on a different IsaacSim/IsaacLab stack, "
              "the replay may play out differently than the original episode.\033[0m")
        return
    for package in ("isaaclab", "isaacsim"):
        recorded_version = recorded.get(f"{package}_version")
        try:
            current_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        if recorded_version is not None and recorded_version != current_version:
            print(f"\033[93mNOTE: this recording was made on {package} {recorded_version} but the "
                  f"current environment runs {package} {current_version}. Recorded outcomes are "
                  "not invariant across simulator versions, so this replay may play out "
                  "differently than the original episode.\033[0m")
