#!/bin/bash
# Render the standard set of preview views of worlds/forest.sdf into build/shots/
cd /mnt/c/ES-proj/camo_ws/scripts
source ../env.sh
bash shoot.sh aerial1 -110 -118 40 0.33 0.82 2>&1 | grep -E "saved|NO FRAME"
bash shoot.sh drone_jeep -50.8 24.9 28 1.0 0 800 600 1.1 2>&1 | grep -E "saved|NO FRAME"
bash shoot.sh drone_red 33.7 38.3 28 1.0 0 800 600 1.1 2>&1 | grep -E "saved|NO FRAME"
bash shoot.sh ground1 -52 14 1.7 0.04 0.9 2>&1 | grep -E "saved|NO FRAME"
