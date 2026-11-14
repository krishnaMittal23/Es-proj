#!/bin/bash
source /opt/ros/jazzy/setup.bash
cd /mnt/c/ES-proj/camo_ws
colcon build --symlink-install --base-paths src --build-base ~/camo_build --install-base ~/camo_install 2>&1 | tail -8
