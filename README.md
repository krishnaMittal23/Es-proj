# Camouflage Detection with a Survey Drone — Gazebo + ROS 2

A quadcopter takes off from a forest helipad, flies an autonomous lawnmower search over a
photo-textured 3D forest (Gazebo Harmonic) and detects **camouflaged soldiers (incl. a ghillie-suit
sniper), a netted vehicle, a camo tent and wildlife** with a YOLOv8 model trained on auto-labelled
simulator data. Detections are geolocated on the terrain and scored against ground truth.

## Run the demo (Windows)
Double-click `run_demo.bat` (or `run_demo.bat baseline` for the classical detector, `run_demo.bat yolo 7`
for another random forest). It opens: Gazebo (chase camera + "AI detections (live)" panel), RViz
(path + found targets) and an annotated live-video window. A mission report prints at the end and
is saved to `results/`.

## The forest world (`scripts/gen_forest.py`)
- 240 m × 240 m hilly terrain with a stream valley, meadows/clearings and a winding dirt trail;
  ground is a 4K texture baked from CC0 photo-scanned textures (Poly Haven) with canopy shadowing.
- ~1,800 procedural trees (oak/beech-like broadleaf, birch, spruce, saplings, dead snags) with real
  bark textures and alpha-cut leaf/needle cards; ~40k grass tufts with wildflowers; ferns, shrubs,
  mossy rocks, stumps and fallen logs (Poly Haven CC0 scans); horizon hills with distant forest.
- Vegetation is merged into 30 m tiles that share textures, so Gazebo runs at real-time factor 1.0.
- Every `--seed` gives a new tree/target layout (used for training worlds vs. the unseen test world).
- Assets: `scripts/fetch_assets.py` downloads the CC0 textures/models into `assets_src/` (not in git).

## Stack
Windows 11 + WSL2 (Ubuntu 24.04, WSLg, NVIDIA GPU via Mesa D3D12) · ROS 2 Jazzy · Gazebo Harmonic ·
PyTorch + Ultralytics YOLOv8n · Python 3.12

## Architecture
```
Gazebo (forest.sdf) --ros_gz_bridge--> /camo/camera/image --> detector_node (YOLO | baseline)
   drone + RGB cam + bbox cam                /camo/odom                  |  /camo/detections
        ^                                                                v
        +------------ /camo/cmd_vel <---------------------------- mission_node
                                   (lawnmower search, 4-frame confirm, terrain ray-cast geolocation, scoring)
```
Files: `src/camo_ros` (nodes + launch), `scripts/` (asset + world generators, training, tests),
`models/` (generated assets), `worlds/forest.sdf`, `runs/camo_yolo` (trained weights + curves).
The original savanna world (`gen_world.py`) is kept for comparison.

## Rebuild from scratch (inside WSL)
`scripts/00_install.sh` → `01_setup_ml.sh` → `02_build.sh` → `python3 scripts/fetch_assets.py` →
`03_collect.sh 300 1 2 3 4 5 6` → `04_train.sh 60` → `06_compare.sh 42`
(`wsl` calls must keep a session open; long jobs die if WSL idles.)

## Results — unseen forest (seed 42, never used for training)
Training data: 1,520 auto-labelled frames from 5 random forests (928 people, 484 vehicles/tents, 376 animals).
YOLOv8n validation: precision 0.96, recall 0.83, mAP50 0.86 (vehicle/tent 0.96, animal 0.86, person 0.77).

| Target | Found | Geolocation error |
|---|---|---|
| Camo soldier standing / crouching | ✓ / ✓ | 0.8 m / 0.5 m |
| Ghillie-suit sniper (prone) | ✓ | 0.3 m |
| Red / blue control persons | ✓ / ✓ | 0.3 m / 0.9 m |
| Jeep under camo netting | ✓ | 0.1 m |
| Bear / deer | ✓ / ✓ | 0.4 m / 1.4 m |
| Camo tent | ✗ | – |

8/9 targets (4/5 camouflaged), 1 false alarm. The original savanna data/model are kept locally in
`dataset_savanna/` and `runs/camo_yolo_savanna/`.

## Limitations (be upfront in the viva)
- Synthetic data only; real-world transfer is not claimed.
- Targets under a closed canopy are invisible from the air (as in reality); targets are placed in
  clearings and at forest edges.
- Drone uses a kinematic velocity controller (no rotor aerodynamics).
