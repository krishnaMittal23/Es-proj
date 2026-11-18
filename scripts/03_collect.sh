#!/bin/bash
# usage: 03_collect.sh [samples_per_world] [seeds...]   (default 350 per world, seeds 1 2 3)
source /mnt/c/ES-proj/camo_ws/env.sh
N=${1:-350}; shift
SEEDS=${@:-1 2 3}
cd /mnt/c/ES-proj/camo_ws/scripts
for S in $SEEDS; do
  echo "=== world seed $S"
  python3 gen_forest.py --seed $S | tail -1
  WORLD=$(python3 -c "import json;print(json.load(open('../worlds/world_info.json'))['world'])")
  cd /tmp
  gz sim -s -r --headless-rendering /mnt/c/ES-proj/camo_ws/worlds/$WORLD.sdf > /tmp/gz_collect.log 2>&1 &
  GZ=$!
  sleep 45
  ros2 run ros_gz_bridge parameter_bridge \
    /camo/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
    /camo/camera/boxes@vision_msgs/msg/Detection2DArray[gz.msgs.AnnotatedAxisAligned2DBox_V \
    /world/$WORLD/set_pose@ros_gz_interfaces/srv/SetEntityPose > /tmp/bridge_collect.log 2>&1 &
  BR=$!
  sleep 4
  timeout 1800 ros2 run camo_ros dataset_node --ros-args -p n_samples:=$N -p tag:=f$S
  kill $BR $GZ 2>/dev/null; sleep 3; pkill -f "gz sim" 2>/dev/null; sleep 2
  cd /mnt/c/ES-proj/camo_ws/scripts
done
echo COLLECT_DONE
