#!/usr/bin/env python3
"""Inject a preview camera into a world: make_preview.py in.sdf out.sdf x y z pitch yaw [w h hfov]
z is metres above terrain (uses forest heightmap)."""
import sys, os, json, numpy as np
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
a = sys.argv
x, y, z, pitch, yaw = map(float, a[3:8])
w, h = (int(a[8]), int(a[9])) if len(a) > 9 else (1280, 720)
fov = float(a[10]) if len(a) > 10 else 1.2
info = json.load(open(os.path.join(WS, "worlds", "world_info.json")))
hm = np.load(os.path.join(WS, "models", info["terrain_dir"], "textures", "heightmap.npy"))
n = hm.shape[0]
u = int(np.clip((x / info["size"] + 0.5) * (n - 1), 0, n - 1)); v = int(np.clip((0.5 - y / info["size"]) * (n - 1), 0, n - 1))
z += float(hm[v, u]) * info["height"] if abs(x) < info["size"] / 2 and abs(y) < info["size"] / 2 else 0
cam = f"""<model name="preview_cam"><static>true</static><pose>{x} {y} {z} 0 {pitch} {yaw}</pose><link name="l"><sensor name="c" type="camera">
<camera><horizontal_fov>{fov}</horizontal_fov><image><width>{w}</width><height>{h}</height></image><clip><near>0.2</near><far>2000</far></clip></camera>
<always_on>1</always_on><update_rate>5</update_rate><topic>/preview/image</topic></sensor></link></model>
"""
s = open(a[1]).read().replace("</world>", cam + "</world>")
s = s.replace("<sensor name=\"camera\"", "<sensor name=\"camera_off\"")
open(a[2], "w").write(s)
