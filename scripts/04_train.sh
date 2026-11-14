#!/bin/bash
source /mnt/c/ES-proj/camo_ws/env.sh
cd /tmp && python3 /mnt/c/ES-proj/camo_ws/scripts/train_yolo.py ${1:-60}
