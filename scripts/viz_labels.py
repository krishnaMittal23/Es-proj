#!/usr/bin/env python3
"""Draw YOLO labels on dataset images -> build/shots/labels_grid.png  (usage: viz_labels.py [prefix] [n])"""
import os, sys, glob, cv2, numpy as np
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pre = sys.argv[1] if len(sys.argv) > 1 else ""
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ims = []
files = sorted(glob.glob(f"{WS}/dataset/images/*/{pre}*.jpg"))
files = [f for f in files if os.path.getsize(f.replace("images", "labels").replace(".jpg", ".txt")) > 0][:n]
for f in files:
    im = cv2.imread(f); h, w = im.shape[:2]
    for line in open(f.replace("images", "labels").replace(".jpg", ".txt")).read().split("\n"):
        if not line.strip():
            continue
        k, cx, cy, bw, bh = map(float, line.split())
        x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), [(0, 255, 255), (255, 160, 0), (0, 0, 255)][int(k)], 2)
    ims.append(cv2.resize(im, (480, 360)))
while len(ims) % 4:
    ims.append(np.zeros_like(ims[0]))
grid = np.concatenate([np.concatenate(ims[i:i + 4], 1) for i in range(0, len(ims), 4)], 0)
cv2.imwrite(f"{WS}/build/shots/labels_grid.png", grid)
print(len(files), "images")
