#!/usr/bin/env python3
"""Build a tiny world to check alpha-cutout support (GLB MASK vs OBJ RGBA)."""
import os, numpy as np
from PIL import Image, ImageDraw
from meshlib import Mesh

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(WS, "build", "alpha_test")
os.makedirs(D, exist_ok=True)
im = Image.new("RGBA", (256, 256), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
rng = np.random.default_rng(0)
for _ in range(40):
    x, y, r = rng.integers(10, 246, 2).tolist() + [int(rng.integers(8, 22))]
    dr.ellipse([x - r, y - r // 2, x + r, y + r // 2], fill=(60, 150, 50, 255))
tp = os.path.join(D, "leaf.png"); im.save(tp)

import shutil
# external-texture GLB + Poly Haven fern (Y-up glTF) as-is
m = Mesh(); m.material("leaf", (1, 1, 1)); m.card("leaf", (0, 0, 0), 3, 3)
md = os.path.join(D, "t_ext", "meshes"); os.makedirs(os.path.join(md, "tex"), exist_ok=True)
im.save(os.path.join(md, "tex", "leaf_ext.png"))
m.write_glb(os.path.join(md, "q.glb"), {"leaf": dict(tex="uri:tex/leaf_ext.png", alpha=True, double=True)})
src = os.path.join(WS, "assets_src", "models", "fern_02")
dst = os.path.join(D, "t_fern", "meshes")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
for nm, uri in (("t_ext", "q.glb"), ("t_fern", "fern_02.gltf")):
    with open(os.path.join(D, nm, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0"?><sdf version="1.9"><model name="{nm}"><static>true</static><link name="l">
<visual name="v"><geometry><mesh><uri>model://{nm}/meshes/{uri}</uri></mesh></geometry></visual></link></model></sdf>""")
    with open(os.path.join(D, nm, "model.config"), "w") as f:
        f.write(f"<?xml version='1.0'?><model><name>{nm}</name><sdf version='1.9'>model.sdf</sdf></model>")

for kind in ("glb", "obj"):
    m = Mesh(); m.material("leaf", (1, 1, 1), "leaf.png")
    m.card("leaf", (0, 0, 0), 3, 3)
    md = os.path.join(D, f"t_{kind}", "meshes")
    if kind == "glb":
        m.write_glb(os.path.join(md, "q.glb"), {"leaf": dict(tex=tp, alpha=True, double=True)})
        uri = "q.glb"
    else:
        m.write(os.path.join(md, "q.obj")); im.save(os.path.join(md, "leaf.png")); uri = "q.obj"
    with open(os.path.join(D, f"t_{kind}", "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0"?><sdf version="1.9"><model name="t_{kind}"><static>true</static><link name="l">
<visual name="v"><geometry><mesh><uri>model://t_{kind}/meshes/{uri}</uri></mesh></geometry></visual></link></model></sdf>""")
    with open(os.path.join(D, f"t_{kind}", "model.config"), "w") as f:
        f.write(f"<?xml version='1.0'?><model><name>t_{kind}</name><sdf version='1.9'>model.sdf</sdf></model>")

with open(os.path.join(D, "world.sdf"), "w") as f:
    f.write("""<?xml version="1.0"?><sdf version="1.9"><world name="alpha">
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
<scene><ambient>0.5 0.5 0.5 1</ambient><background>0.6 0.75 0.95 1</background><sky/></scene>
<light type="directional" name="sun"><cast_shadows>true</cast_shadows><diffuse>1 1 1 1</diffuse><direction>-0.5 0.3 -0.8</direction></light>
<model name="ground"><static>true</static><link name="l"><visual name="v"><geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
<material><diffuse>0.7 0.2 0.2 1</diffuse><ambient>0.7 0.2 0.2 1</ambient></material></visual></link></model>
<include><uri>model://t_glb</uri><pose>6 -2 0 0 0 0</pose></include>
<include><uri>model://t_ext</uri><pose>6 2 0 0 0 0</pose></include>
<include><uri>model://t_fern</uri><pose>4 0 0 0 0 0</pose></include>
<model name="cam"><static>true</static><pose>0 0 1.5 0 0 0</pose><link name="l"><sensor name="c" type="camera">
<camera><horizontal_fov>1.2</horizontal_fov><image><width>800</width><height>500</height></image><clip><near>0.1</near><far>500</far></clip></camera>
<always_on>1</always_on><update_rate>5</update_rate><topic>/preview/image</topic></sensor></link></model>
</world></sdf>""")
print("ok", D)
