#!/bin/bash
# make the Gazebo GUI chase-cam follow the drone (retries until the GUI is up)
source /mnt/c/ES-proj/camo_ws/env.sh
for i in $(seq 1 30); do
  gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req 'data: "camo_drone"' 2>/dev/null | grep -q true && break
  sleep 3
done
gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 2000 --req 'x: -14, y: 0, z: 8' 2>/dev/null
echo CHASE_CAM_READY
