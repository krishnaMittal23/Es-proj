#!/bin/bash
source /mnt/c/ES-proj/camo_ws/env.sh
cd /tmp
gz sim -s -r --headless-rendering /mnt/c/ES-proj/camo_ws/worlds/savanna.sdf > /tmp/gzsim.log 2>&1 &
GZ=$!
sleep 20
ros2 run ros_gz_bridge parameter_bridge \
 /camo/camera/image@sensor_msgs/msg/Image[gz.msgs.Image \
 /camo/camera/boxes@vision_msgs/msg/Detection2DArray[gz.msgs.AnnotatedAxisAligned2DBox_V \
 /camo/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry \
 /camo/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
 /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock > /tmp/bridge.log 2>&1 &
sleep 6
head -20 /tmp/bridge.log
ros2 topic list
timeout 8 ros2 topic echo --once /camo/odom --field pose.pose.position
timeout 8 ros2 topic hz /camo/camera/image -w 5 2>&1 | head -3
pkill -f parameter_bridge; kill $GZ
