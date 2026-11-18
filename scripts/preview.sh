#!/bin/bash
# Render a world headless and save a frame from /preview/image.
# usage: preview.sh <world.sdf> <out.png> [extra GZ_SIM_RESOURCE_PATH] [warmup_s]
source /mnt/c/ES-proj/camo_ws/env.sh
export GZ_SIM_RESOURCE_PATH=${3:+$3:}$GZ_SIM_RESOURCE_PATH
cd /tmp
timeout 300 gz sim -s -r --headless-rendering "$1" > /tmp/gz_preview.log 2>&1 &
GZ=$!
sleep ${4:-15}
timeout 200 ros2 run ros_gz_bridge parameter_bridge /preview/image@sensor_msgs/msg/Image[gz.msgs.Image > /tmp/bridge_preview.log 2>&1 &
BR=$!
timeout 150 python3 /mnt/c/ES-proj/camo_ws/scripts/snap.py "$2" /preview/image 3
grep -iE "error|warn" /tmp/gz_preview.log | grep -v "Gtk" | sort | uniq -c | head -8
kill $GZ $BR 2>/dev/null; sleep 1; kill -9 $GZ $BR 2>/dev/null
pkill -9 -f "gz sim -s" 2>/dev/null; true
