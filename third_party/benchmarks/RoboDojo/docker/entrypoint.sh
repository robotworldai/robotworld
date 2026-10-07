#!/usr/bin/env bash
# Activate the RoboDojo conda env, then exec the requested command from the
# RoboDojo project root. This is the image ENTRYPOINT, so `docker run ... <cmd>`
# runs <cmd> inside the activated env (default <cmd> is an interactive bash).
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo

cd /workspace/RoboDojo

exec "$@"
