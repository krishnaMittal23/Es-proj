# Viva demo script (~6 min)
1. **Problem (30 s):** camouflaged people/vehicles/animals are hard to spot from the air.
2. **Show the world (45 s):** Gazebo overview — terrain, acacia/baobab, hidden targets. Mention 9 hidden targets; some camouflaged, one control group in plain colours.
3. **Run:** double-click `run_demo.bat`. Explain the three windows while the drone takes off:
   Gazebo chase-cam, annotated camera feed, RViz (yellow path, red spheres = confirmed targets).
4. **While it flies (2–3 min):** pipeline = Gazebo sensors → ros_gz_bridge → YOLO node → mission node.
   Point out: a target is only reported after 4 consistent frames; position comes from ray-casting the box
   onto the terrain heightmap.
5. **Report:** read the mission report (found / false alarms / camouflaged split).
6. **Contrast (1 min):** show the results table: classical baseline gave 287 false alarms, YOLO 1.
7. **Limitations + future work:** real-data transfer, thermal/multispectral cameras, real PX4 flight dynamics.
Backup: if Gazebo is slow, run `run_demo.bat` once beforehand and show `results/mission_report.json`.
