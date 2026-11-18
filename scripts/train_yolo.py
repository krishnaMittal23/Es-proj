#!/usr/bin/env python3
"""Fine-tune YOLOv8n on the Gazebo camouflage dataset."""
import os, shutil, sys
from ultralytics import YOLO
WS = os.environ.get("CAMO_WS", "/mnt/c/ES-proj/camo_ws")
LOCAL = os.path.expanduser("~/camo_data")
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
if os.path.exists(LOCAL): shutil.rmtree(LOCAL)
shutil.copytree(os.path.join(WS, "dataset"), LOCAL)
open(os.path.join(LOCAL, "data.yaml"), "w").write(
    f"path: {LOCAL}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: vehicle/tent\n  2: animal\n")
m = YOLO("yolov8n.pt")
m.train(data=os.path.join(LOCAL, "data.yaml"), epochs=epochs, imgsz=800, batch=8, device=0, workers=2,
        project=os.path.join(WS, "runs"), name="camo_yolo", exist_ok=True, patience=20,
        degrees=0, flipud=0.5, fliplr=0.5, mosaic=1.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, plots=True)
r = m.val()
print("mAP50", r.box.map50, "mAP50-95", r.box.map)
