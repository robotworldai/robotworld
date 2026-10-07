#!/bin/bash
set -euo pipefail

# Install XPolicyLab plus the vendored OpenWAM package into the active
# environment. Install PyTorch first (see the policy README) — this script
# does not pin a CUDA wheel.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPENWAM_ROOT="${OPENWAM_ROOT:-${SCRIPT_DIR}/OpenWAM}"

python -m pip install -e "${XPL_ROOT}"
python -m pip install -e "${OPENWAM_ROOT}"
