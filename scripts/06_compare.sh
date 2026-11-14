#!/bin/bash
# runs YOLO then baseline on the same unseen world and prints both reports
for M in yolo baseline; do
  touch /tmp/start_marker
  bash /mnt/c/ES-proj/camo_ws/scripts/05_mission_test.sh $M ${1:-42} > /tmp/run_$M.out 2>&1
  echo "######## $M"; grep -E "TARGET|targets in|detected|false alarms|recall|precision|camouflaged|plain/|flight|Traceback|Error|detector mode" /tmp/mission.log | cut -c35-200
  sleep 5
done
