#!/bin/bash

# * Set all the variables here
IMAGE_NAME="isaac-lab-anlun:2.1.0"
CONTAINER_NAME="isaac-lab-anlun"

# The directories need to be mounted into workspace
CODE_DIRS=(
    "$HOME/g1-humanoid"
)
# * End of variables

# Make sure the directories exist
for dir in "${CODE_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist. Please create it first."
        exit 1
    fi
done

# Build volume mount and PYTHONPATH strings
dir_mounts=""
pythonpath=""
for dir in "${CODE_DIRS[@]}"; do
    base=$(basename "$dir")
    dir_mounts+="-v $dir:/workspace/$base:rw "
    # Add to PYTHONPATH if source exists
    if [ -d "$dir/source" ]; then
        pythonpath+="/workspace/$base/source:"
    fi
done
# Remove trailing colon from PYTHONPATH
pythonpath=${pythonpath%:}

# Pull and tag image
docker pull nvcr.io/nvidia/isaac-lab:2.1.0
docker tag nvcr.io/nvidia/isaac-lab:2.1.0 $IMAGE_NAME

docker run --name "$CONTAINER_NAME" --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v $HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v $HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v $HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v $HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v $HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v $HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v $HOME/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v $HOME/docker/isaac-sim/documents:/root/Documents:rw \
   $dir_mounts \
   -e PYTHONPATH=$PYTHONPATH:$pythonpath \
   $IMAGE_NAME