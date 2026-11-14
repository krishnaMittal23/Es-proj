#!/bin/bash
source /mnt/c/ES-proj/camo_ws/env.sh
cd /tmp
gz sim -s -r --headless-rendering /mnt/c/ES-proj/camo_ws/worlds/savanna.sdf > /tmp/gzsim.log 2>&1 &
GZ=$!
sleep 20
gz topic -l | grep -iE "camo|cmd_vel"
pos() { timeout 5 gz topic -e -n 1 -t /camo/odom | grep -A3 "position {" | tr -d '\n '; echo; }
echo before; pos
for i in $(seq 1 12); do gz topic -t /model/camo_drone/cmd_vel -m gz.msgs.Twist -p "linear:{x:5.0}"; sleep 0.25; done
echo after_forward; pos
for i in $(seq 1 8); do gz topic -t /model/camo_drone/cmd_vel -m gz.msgs.Twist -p "angular:{z:1.0}"; sleep 0.25; done
echo after_yaw; timeout 5 gz topic -e -n 1 -t /camo/odom | grep -A4 "orientation {" | tr -d '\n '; echo
for i in $(seq 1 12); do gz topic -t /model/camo_drone/cmd_vel -m gz.msgs.Twist -p "linear:{x:5.0}"; sleep 0.25; done
echo after_forward_after_yaw; pos
kill $GZ
