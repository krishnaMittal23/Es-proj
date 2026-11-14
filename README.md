# Camouflage Detection with a Survey Drone — Gazebo + ROS 2

A quadcopter flies an autonomous lawnmower search over a procedurally generated
African-savanna world (Gazebo Harmonic) and detects **camouflaged people, vehicles, tents and
animals** with a YOLOv8 model trained on auto-labelled simulator data. Detections are
geolocated on the terrain and scored against ground truth.

## Run the demo (Windows)
Double-click `run_demo.bat` (or `run_demo.bat baseline` for the classical detector).
It opens: Gazebo (chase camera + drone-camera panel), RViz (path + found targets),
and an annotated live-video window. A mission report prints at the end and is saved to `results/`.

## Stack
Windows 11 + WSL2 (Ubuntu 24.04, WSLg, NVIDIA GPU via Mesa D3D12) · ROS 2 Jazzy · Gazebo Harmonic ·
PyTorch + Ultralytics YOLOv8n · Python 3.12

## Architecture
```
Gazebo (savanna.sdf) --ros_gz_bridge--> /camo/camera/image --> detector_node (YOLO | baseline)
   drone + RGB cam + bbox cam                 /camo/odom                 |  /camo/detections
        ^                                                                v
        +------------ /camo/cmd_vel <---------------------------- mission_node
                                   (lawnmower search, 4-frame confirm, terrain ray-cast geolocation, scoring)
```
Packages / files: `src/camo_ros` (nodes + launch), `scripts/` (asset+world generators, training, tests),
`models/` (generated assets), `worlds/savanna.sdf`, `runs/camo_yolo` (trained weights + curves).

## Rebuild from scratch (inside WSL)
`scripts/00_install.sh` → `01_setup_ml.sh` → `02_build.sh` → `python3 gen_world.py` →
`03_collect.sh 350 1 2 3` → `04_train.sh 60` → `06_compare.sh 42`
(`wsl` calls must keep a session open; long jobs die if WSL idles.)

## Results — unseen world (seed 42, never used for training)
| Detector | Found | Camouflaged | False alarms | Precision |
|---|---|---|---|---|
| YOLOv8n (trained) | 9/9 | 5/5 | 1 | 90% |
| Classical colour-anomaly baseline | 8/9 | 4/5 | 287 | 3% |
YOLO validation: mAP50 0.96, ~7 ms/frame on RTX 2050. Geolocation error < 2 m.
Different seeds give new layouts: `run_demo.bat yolo 7`.

## Limitations (be upfront in the viva)
- Synthetic data only; textures are procedural, so real-world transfer is not claimed.
- Validation split shares worlds with training; the seed-42 mission is the honest test.
- Drone uses a kinematic velocity controller (no rotor aerodynamics).
- One run per detector so far; precision varies a little between runs.
