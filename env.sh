# source this: source /mnt/c/ES-proj/camo_ws/env.sh
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export CAMO_WS=/mnt/c/ES-proj/camo_ws
export GZ_SIM_RESOURCE_PATH=$CAMO_WS/models:$CAMO_WS/worlds:$GZ_SIM_RESOURCE_PATH
export QT_QPA_PLATFORM=xcb
export SDF_PATH=$CAMO_WS/models:$SDF_PATH
export GZ_FILE_PATH=$CAMO_WS/models:$GZ_FILE_PATH
# python venv with ultralytics (created by scripts/01_setup_ml.sh)
if [ -d "$HOME/camo_venv" ]; then export PYTHONPATH=$HOME/camo_venv/lib/python3.12/site-packages:$PYTHONPATH; fi
[ -f "$HOME/camo_install/setup.bash" ] && source "$HOME/camo_install/setup.bash"
