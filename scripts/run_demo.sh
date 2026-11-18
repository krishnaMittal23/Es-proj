#!/bin/bash
# usage: run_demo.sh [mode=yolo|baseline] [seed=42]
source /mnt/c/ES-proj/camo_ws/env.sh
MODE=${1:-yolo}; SEED=${2:-42}
cd /mnt/c/ES-proj/camo_ws/scripts
if [ "$(python3 -c "import json;print(json.load(open('../worlds/targets.json')).get('seed'))" 2>/dev/null)" != "$SEED" ] \
   || [ ! -f ../worlds/forest.sdf ]; then
  python3 gen_forest.py --seed $SEED
fi
cd /tmp && ros2 launch camo_ros demo.launch.py gui:=true rviz:=true mode:=$MODE
