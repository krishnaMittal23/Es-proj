# Camouflage Detection with a Survey Drone — Gazebo + ROS 2 + YOLOv8

A simulated quadcopter takes off from a forest helipad, flies an autonomous **lawnmower search** over a
photo-textured 3D forest and finds **camouflaged soldiers (including a ghillie-suit sniper), a netted jeep,
a camo tent and wildlife**. A YOLOv8 model trained on auto-labelled simulator data does the detecting;
detections are **geolocated on the terrain** and **scored against hidden ground truth**.

> **Simulate. Learn. Search. Score.**
> A generated forest world → a detector trained on free, automatic labels → an autonomous search mission →
> a mission report (found / false alarms / recall / precision / position error).

A long, plain-language study guide (architecture, every file, every parameter, viva Q&A) is in
[`Camouflage_Detection_Project_Guide.pdf`](Camouflage_Detection_Project_Guide.pdf).

---

## Contents
1. [Features](#features)
2. [How it works](#how-it-works)
3. [Results](#results)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [Run the demo](#run-the-demo)
7. [The forest world](#the-forest-world)
8. [Training your own detector](#training-your-own-detector)
9. [Mission parameters](#mission-parameters)
10. [Project layout](#project-layout)
11. [Troubleshooting](#troubleshooting)
12. [Limitations and future work](#limitations-and-future-work)
13. [Credits and licences](#credits-and-licences)

---

## Features
- **Procedural forest world** (`scripts/gen_forest.py`): 240 m × 240 m hilly terrain with a stream valley, meadows and a
  winding trail; about 1,800 trees (broadleaf, birch, spruce, saplings, snags), about 40,000 grass tufts with wildflowers,
  ferns, shrubs, mossy rocks, stumps and logs. Every `--seed` gives a new layout.
- **Nine hidden targets:** three camouflaged people (standing, crouching, prone ghillie sniper), a camouflaged jeep under
  netting, a camo tent, a deer, a bear, and two plainly dressed control persons (red and blue).
- **Auto-labelled dataset collection:** a bounding-box camera in the simulator gives exact boxes, so thousands of YOLO
  training images are produced with no manual labelling.
- **Two detectors:** fine-tuned YOLOv8n, and a classical colour-anomaly baseline that shows why camouflage is hard.
- **Autonomous mission:** take-off, terrain-following lawnmower search, 4-frame confirmation of each target,
  ray-cast geolocation onto the terrain height map, return to launch, scoring against ground truth.
- **Live visualisation:** Gazebo chase camera, an annotated live video window, and RViz (flight path + found-target markers).

## How it works

```
Gazebo (forest.sdf) --ros_gz_bridge--> /camo/camera/image --> detector_node (YOLO | baseline)
   drone + RGB cam + bbox cam                /camo/odom                  |  /camo/detections
        ^                                                                v
        +------------ /camo/cmd_vel <---------------------------- mission_node
                                   (lawnmower search, 4-frame confirm, terrain ray-cast geolocation, scoring)
```

| Node | Job |
|---|---|
| `detector_node` | Runs YOLOv8 (or the baseline) on each camera frame; publishes boxes and an annotated image |
| `mission_node` | State machine: TAKEOFF → SEARCH ⇄ CONFIRM → RTL → DONE; geolocates, confirms and scores targets |
| `viewer_node` | OpenCV window with the annotated feed and a status bar |
| `dataset_node` | Teleports the drone around targets and saves images + YOLO labels (training data only) |

**Geolocation:** the bottom-centre pixel of a box is turned into a ray (pinhole camera, 57° down-tilt, drone heading) and
marched forward until it passes below the terrain height map. **Confirmation:** a target is only reported after 4
consistent detections (score ≥ 0.45) near the same ground position. **Scoring:** a confirmed target within 6 m of a true
target is a true positive; everything else is a false alarm.

## Results

Evaluated on an **unseen forest (seed 42, never used for training)**.

Training data: 1,520 auto-labelled frames from 5 random forests (928 people, 484 vehicles/tents, 376 animals).
YOLOv8n validation: precision 0.96, recall 0.83, mAP50 0.86 (vehicle/tent 0.96, animal 0.86, person 0.77).
A re-run of the training on an RTX 2050 laptop (60 epochs, 26 minutes) reached mAP50 0.854.

| Target | Found | Geolocation error |
|---|---|---|
| Camo soldier standing / crouching | ✓ / ✓ | 0.8 m / 0.5 m |
| Ghillie-suit sniper (prone) | ✓ | 0.3 m |
| Red / blue control persons | ✓ / ✓ | 0.3 m / 0.9 m |
| Jeep under camo netting | ✓ | 0.1 m |
| Bear / deer | ✓ / ✓ | 0.4 m / 1.4 m |
| Camo tent | ✗ | – |

8/9 targets (4/5 camouflaged), 1 false alarm. A full mission report is written to `results/mission_report.json`.

## Requirements

| | |
|---|---|
| OS | Windows 11 with **WSL2** and **WSLg** (Linux GUI windows) |
| Linux | Ubuntu 24.04 in WSL2 |
| Simulator / robotics | Gazebo Harmonic, ROS 2 Jazzy (`ros-jazzy-desktop`, `ros-jazzy-ros-gz`) |
| Python | 3.12, PyTorch (CUDA build), Ultralytics YOLOv8, OpenCV, NumPy < 2, SciPy, Pillow |
| GPU | NVIDIA GPU recommended (tested on an RTX 2050, 4 GB). The forest is about 4.7 M triangles |
| Disk | About 20 GB for the Linux install plus about 1 GB for generated data |
| RAM | 16 GB recommended |

## Setup

> **Path note.** The scripts expect the project at `/mnt/c/ES-proj/camo_ws` (that is `C:\ES-proj\camo_ws` on Windows).
> If your folder has a different name, make a Windows junction so no code needs changing:
> `cmd /c mklink /J C:\ES-proj\camo_ws C:\path\to\this\project`

1. **Install WSL2 + Ubuntu 24.04** (PowerShell): `wsl --install -d Ubuntu-24.04`, then create a user.
2. **Install ROS 2 + Gazebo** (inside Ubuntu): `bash scripts/00_install.sh`
3. **Install the ML stack** (PyTorch + YOLO in `~/camo_venv`): `bash scripts/01_setup_ml.sh`
4. **Build the ROS package:** `bash scripts/02_build.sh` (build output goes to `~/camo_build` and `~/camo_install`)
5. **Get the assets** (CC0 textures and models from Poly Haven): `python3 scripts/fetch_assets.py`
   — or copy an existing `assets_src/` folder into the project root.
6. **Get the detector.** Either copy a trained `runs/camo_yolo/weights/best.pt`, or train one
   (see [Training your own detector](#training-your-own-detector)). Without weights the detector falls back to the baseline.

Line endings: the `.sh` files must use Unix (LF) line endings. If bash complains about `\r`, run
`sed -i 's/\r$//' scripts/*.sh env.sh`.

Behind a proxy (for example a college network)? Configure it **inside WSL** for apt, curl, wget and pip. If HTTPS
through the proxy hangs while HTTP works, lower the WSL MTU (`sudo ip link set dev eth0 mtu 1200`).

## Run the demo

Double-click `run_demo.bat`, or from a terminal:

```
run_demo.bat                 # YOLO detector, forest seed 42
run_demo.bat baseline        # classical colour-anomaly detector
run_demo.bat yolo 7          # YOLO on a different random forest (seed 7)
```

It opens Gazebo (chase camera), the "Camouflage Detection – Drone Camera" video window and RViz. The mission takes
roughly 4 minutes of simulated time; a report prints at the end and is saved to `results/`.

Headless (no windows) run and comparison on the unseen forest:

```
bash scripts/05_mission_test.sh yolo 42     # one headless mission -> results/report_yolo_seed42.json
bash scripts/06_compare.sh 42               # YOLO then baseline, both reports printed
```

Launch file options (`ros2 launch camo_ros demo.launch.py ...`): `mode:=yolo|baseline`, `gui:=true|false`,
`rviz:=true|false`, `detector`, `viewer`, `autostart`, `conf:=0.35`.

To stop everything: `wsl --terminate Ubuntu-24.04`.

## The forest world

`python3 scripts/gen_forest.py [--seed N] [--targets K]` (about 50 s) writes the terrain, vegetation tiles, target models,
`worlds/forest.sdf`, `worlds/targets.json` (ground truth) and `worlds/world_info.json`.

- Vegetation is merged into 30 m tiles that share textures so Gazebo can run near real time.
- The same seed always reproduces the same forest; different seeds make new training and test worlds.
- Assets: `scripts/fetch_assets.py` downloads CC0 textures and models into `assets_src/` (not in git).
- The original savanna world (`gen_world.py`) is kept for comparison.

## Training your own detector

```
bash scripts/03_collect.sh 300 1 2 3 4 5 6      # 300 labelled frames from each of several forests
bash scripts/04_train.sh 60                     # fine-tune YOLOv8n, up to 60 epochs
```

- `dataset_node` flies the drone near random targets (altitude 22–36 m, 15% empty "negative" views) and saves JPEGs plus
  YOLO labels into `dataset/` (85% train, 15% validation).
- Classes: `0 person`, `1 vehicle/tent`, `2 animal`. Camouflaged and plain people share a class.
- Training uses `yolov8n.pt` as the starting point (downloaded on first run), image size 800, batch 8, flip and
  mosaic augmentation, and writes weights and charts to `runs/camo_yolo/`.
- Keep a forest seed out of training (the demo uses seed 42) so the evaluation stays honest.

## Mission parameters

Set as ROS parameters on `mission_node` (defaults in `src/camo_ros/camo_ros/mission_node.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `altitude_agl` | 28.0 m | Height above ground (clears the roughly 20 m canopy) |
| `speed` | 6.0 m/s | Cruise speed |
| `lane_spacing` | 22.0 m | Distance between search lanes (7 lanes, 14 waypoints) |
| `area` | 75.0 m | Half-size of the square search area |
| `min_conf` | 0.45 | Minimum detection score used by the mission |
| `confirm_frames` | 4 | Consistent detections needed to confirm a target |
| `dedupe_radius` | 5.0 m | Ignore detections this close to confirmed targets |
| `match_radius` | 6.0 m | Max error for a true positive when scoring |

The detector node has its own threshold, `conf` (0.35), `imgsz` (800) and `weights` (default `runs/camo_yolo/weights/best.pt`).

## Project layout

```
README.md, DEMO_SCRIPT.md        this file; 6-minute viva demo script
Camouflage_Detection_Project_Guide.pdf   detailed study guide
run_demo.bat                     Windows launcher
env.sh                           ROS 2 / Gazebo / GPU environment (sourced by the scripts)
config/camo.rviz                 RViz layout
src/camo_ros/                    ROS 2 package: detector_node, mission_node, viewer_node, dataset_node, launch file
scripts/                         install, world generators, data collection, training, tests, preview tools
models/                          drone and target models (generated forest tiles are not in git)
worlds/                          forest.sdf, targets.json (ground truth), world_info.json
dataset/  runs/  results/        generated: training data, weights and charts, mission reports (not in git)
assets_src/                      downloaded Poly Haven textures and models (not in git)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `invalid option` / `$'\r'` errors in scripts | Convert line endings: `sed -i 's/\r$//' scripts/*.sh env.sh` |
| `/mnt/c/ES-proj/camo_ws: No such file` | Create the `C:\ES-proj\camo_ws` junction (see Setup) |
| Detector says "weights missing -> falling back to baseline" | Provide or train `runs/camo_yolo/weights/best.pt` |
| Downloads hang inside WSL | Configure the proxy in WSL; try `mtu 1200` |
| Gazebo or the demo is slow | Plug in, set Windows to Best performance, set `wsl.exe` and `wslg.exe` to High performance in Settings → Graphics, close RViz and other heavy apps, keep the Gazebo window small |
| No windows appear | Check WSLg: `echo $DISPLAY` should print `:0` inside WSL |
| First training run cannot download `yolov8n.pt` | Check the network or proxy; place the file in the working directory |

## Limitations and future work
- **It is a simulation.** A detector trained only on Gazebo renders will usually do worse on real photographs (sim-to-real gap).
- **The drone has no real flight dynamics** (gravity off, velocity-controlled, never tilts); the geolocation assumes a level camera.
- **Single test world:** the headline numbers come from one unseen forest with nine targets.
- Scoring matches by position only, not by class.
- Future work: thermal and multispectral cameras, PX4 software-in-the-loop flight, more varied training worlds,
  testing on real aerial data, a bigger model and a tracker, adaptive search that circles suspicious spots.

## Credits and licences
- Ground, bark, grass, fern, shrub and rock assets: [Poly Haven](https://polyhaven.com), **CC0**.
- Detection model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0; check its terms before any
  commercial use).
- Built with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) and [Gazebo Harmonic](https://gazebosim.org/).
- The `camo_ros` package declares an MIT licence in `package.xml`.
