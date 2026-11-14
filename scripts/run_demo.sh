#!/bin/bash
# usage: run_demo.sh [mode=yolo|baseline] [seed=42]
source /mnt/c/ES-proj/camo_ws/env.sh
MODE=${1:-yolo}; SEED=${2:-42}
cd /mnt/c/ES-proj/camo_ws/scripts && python3 gen_world.py --seed $SEED --skip-assets
cd /tmp && ros2 launch camo_ros demo.launch.py gui:=true rviz:=true mode:=$MODE
