#!/bin/bash
# Close-up renders of a few targets (from worlds/targets.json) into build/shots/tgt_*.png
cd /mnt/c/ES-proj/camo_ws/scripts
source ../env.sh
python3 - <<'EOF' > /tmp/tgt_poses.txt
import json
for t in json.load(open("/mnt/c/ES-proj/camo_ws/worlds/targets.json"))["targets"]:
    print(t["model"], t["x"] - 6, t["y"] - 6)
EOF
while read name x y; do
  case "$name" in person_plain_stand|person_ghillie_prone|jeep_camo_net|deer|person_camo_crouch|tent_camo)
    bash shoot.sh tgt_$name $x $y 3.2 0.33 0.785 960 540 0.9 2>&1 | grep -E "saved|NO FRAME";;
  esac
done < /tmp/tgt_poses.txt
