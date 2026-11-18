# Viva demo script (~6 min)
1. **Problem (30 s):** camouflaged people/vehicles/animals are hard to spot from the air.
2. **Show the world (45 s):** Gazebo overview — hilly forest (~1,800 trees: oak, birch, spruce), meadows,
   stream and trail, photo-scanned ground textures. Mention 9 hidden targets: woodland-camo soldiers,
   a ghillie-suit sniper, a netted jeep, a camo tent, deer and bear, plus two plainly dressed control persons.
3. **Run:** double-click `run_demo.bat`. Explain the three windows while the drone takes off:
   Gazebo chase-cam, annotated camera feed, RViz (yellow path, red spheres = confirmed targets).
4. **While it flies (2–3 min):** pipeline = Gazebo sensors → ros_gz_bridge → YOLO node → mission node.
   Point out: a target is only reported after 4 consistent frames; position comes from ray-casting the box
   onto the terrain heightmap.
5. **Report:** read the mission report (found / false alarms / camouflaged split).
6. **Contrast (1 min):** show the README results table: on an unseen forest YOLO found 8/9 targets
   (including the ghillie sniper) with geolocation error under 1.5 m and 1 false alarm.
   `run_demo.bat baseline` shows why a classical colour-anomaly detector fails on camouflage.
7. **Limitations + future work:** real-data transfer, thermal/multispectral cameras, real PX4 flight dynamics.
Backup: if Gazebo is slow, run `run_demo.bat` once beforehand and show `results/mission_report.json`.
