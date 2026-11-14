#!/bin/bash
# headless full mission on unseen world; usage: 05_mission_test.sh [mode] [seed]
source /mnt/c/ES-proj/camo_ws/env.sh
MODE=${1:-yolo}; SEED=${2:-42}
cd /mnt/c/ES-proj/camo_ws/scripts && python3 gen_world.py --seed $SEED --skip-assets
cd /tmp
timeout 900 ros2 launch camo_ros demo.launch.py gui:=false viewer:=false mode:=$MODE > /tmp/mission.log 2>&1 &
LP=$!
for i in $(seq 1 120); do [ /mnt/c/ES-proj/camo_ws/results/mission_report.json -nt /tmp/start_marker ] && break; sleep 5; done
cp /mnt/c/ES-proj/camo_ws/results/mission_report.json /mnt/c/ES-proj/camo_ws/results/report_${MODE}_seed${SEED}.json 2>/dev/null
sleep 3; pkill -f "ros2 launch"; pkill -f "gz sim"; pkill -f parameter_bridge; pkill -f _node; wait 2>/dev/null
