#!/bin/bash
# usage: shoot.sh out_name x y z_agl pitch yaw [w h fov] ; renders worlds/forest.sdf from that pose
WS=/mnt/c/ES-proj/camo_ws
OUT=$WS/build/shots; mkdir -p $OUT
python3 $WS/scripts/make_preview.py $WS/worlds/forest.sdf /tmp/preview_world.sdf "${@:2}"
bash $WS/scripts/preview.sh /tmp/preview_world.sdf $OUT/$1.png "" ${WARMUP:-45}
