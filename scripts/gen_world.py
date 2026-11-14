#!/usr/bin/env python3
"""Generate worlds/savanna.sdf (terrain, vegetation, camouflaged targets, drone).

usage: gen_world.py [--seed N] [--targets K]
Writes worlds/savanna.sdf and worlds/targets.json (ground truth for scoring).
"""
import os, json, math, argparse
import numpy as np
import gen_assets as A

WS = A.WS
HALF = A.TERRAIN_SIZE / 2


def inc(name, uri, x, y, z, yaw=0.0, extra=""):
    return f"""    <include><name>{name}</name><uri>model://{uri}</uri><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.3f}</pose>{extra}</include>\n"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", type=int, default=9)
    ap.add_argument("--skip-assets", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    T = A.build_all() if not args.skip_assets else A.Terrain()
    body = ""
    placed = []  # (x,y,r) for spacing

    def free(x, y, r):
        return all((x - px) ** 2 + (y - py) ** 2 > (r + pr) ** 2 for px, py, pr in placed)

    def pick(rmin, rmax, avoid_river=True, area=(-95, 95), maxslope=0.35, tries=200):
        for _ in range(tries):
            x, y = rng.uniform(*area, 2)
            if avoid_river and T.riverness(x, y) > 0.35:
                continue
            e = 1.5
            sl = max(abs(T.z(x + e, y) - T.z(x - e, y)), abs(T.z(x, y + e) - T.z(x, y - e))) / (2 * e)
            if sl > maxslope:
                continue
            if free(x, y, rmax):
                placed.append((x, y, rmax))
                return x, y
        return None

    # scenery ---------------------------------------------------------
    n = 0
    def scatter(prefix, variants, count, r, zoff=-0.05, avoid_river=True, area=(-110, 110), scale_rng=(1, 1)):
        nonlocal body, n
        for _ in range(count):
            p = pick(r, r, avoid_river, area)
            if not p:
                continue
            x, y = p
            v = int(rng.integers(0, variants))
            n += 1
            mname = prefix if prefix in ("termite_mound", "fallen_log") else f"{prefix}_{v}"
            body += inc(f"{prefix}_{n}", mname, x, y, T.z(x, y) + zoff, rng.uniform(0, 6.28))

    scatter("acacia", 4, 42, 5.0)
    scatter("baobab", 2, 7, 5.0)
    scatter("bush", 3, 90, 1.6)
    scatter("drybush", 2, 70, 1.6)
    scatter("rock", 3, 50, 1.4, avoid_river=False)
    scatter("grass_clump", 2, 160, 0.7, avoid_river=False)
    scatter("termite_mound", 1, 14, 1.0)
    scatter("fallen_log", 1, 10, 2.0)

    # camouflaged targets, hidden next to cover -----------------------
    types = [
        ("person_camo_stand", 1.0, "person_camo"),
        ("person_camo_crouch", 1.0, "person_camo"),
        ("person_camo_prone", 1.0, "person_camo"),
        ("person_plain_stand", 1.0, "person_plain"),   # control: NOT camouflaged
        ("person_plain_prone", 1.0, "person_plain"),
        ("tent_camo", 2.5, "tent_camo"),
        ("jeep_camo", 3.0, "jeep_camo"),
        ("rhino", 2.5, "animal"),
        ("hippo", 2.2, "animal"),
    ]
    targets = []
    tpl = list(types)
    for i in range(args.targets):
        name, r, cls = tpl[i % len(tpl)]
        p = pick(r, r + 1.0, area=(-70, 70), maxslope=0.25)
        if not p:
            continue
        x, y = p
        yaw = float(rng.uniform(0, 6.28))
        z = T.z(x, y) - 0.02
        body += inc(f"target_{i}_{name}", name, x, y, z, yaw)
        targets.append(dict(id=i, model=name, cls=cls, x=round(x, 2), y=round(y, 2), z=round(z, 2),
                            camouflaged=("plain" not in name and name not in ("rhino", "hippo"))))
        # put cover beside each target (bush) so it is *partly* hidden
        if rng.random() < 0.8:
            bx, by = x + rng.uniform(1.0, 2.2) * rng.choice([-1, 1]), y + rng.uniform(1.0, 2.2) * rng.choice([-1, 1])
            body += inc(f"cover_{i}", f"bush_{int(rng.integers(0, 3))}", bx, by, T.z(bx, by) - 0.05, rng.uniform(0, 6.28))

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="savanna">
    <physics name="1ms" type="ode"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>

    <gravity>0 0 -9.8</gravity>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model><latitude_deg>-2.3333</latitude_deg><longitude_deg>34.8333</longitude_deg>
      <elevation>1500</elevation><heading_deg>0</heading_deg>
    </spherical_coordinates>

    <scene>
      <ambient>0.55 0.52 0.48 1</ambient>
      <background>0.62 0.78 0.95 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
      <sky/>
    </scene>

    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui><title>3D View</title><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui>
        <engine>ogre2</engine><scene>scene</scene><ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.62 0.78 0.95</background_color><camera_pose>-125 -125 55 0 0.38 0.785</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager"><gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control"><gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="CameraTracking" name="Camera Tracking"><gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="SelectEntities" name="Select Entities"><gz-gui><anchors target="Select entities"><line own="right" target="right"/><line own="top" target="top"/></anchors><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui><title>World control</title><property type="bool" key="showTitleBar">false</property><property type="bool" key="resizable">false</property><property type="double" key="height">72</property><property type="double" key="width">121</property><property type="double" key="z">1</property><property type="string" key="state">floating</property><anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors></gz-gui>
        <play_pause>true</play_pause><step>true</step><start_paused>false</start_paused>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui><title>World stats</title><property type="bool" key="showTitleBar">false</property><property type="bool" key="resizable">false</property><property type="double" key="height">110</property><property type="double" key="width">290</property><property type="double" key="z">1</property><property type="string" key="state">floating</property><anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors></gz-gui>
        <sim_time>true</sim_time><real_time>true</real_time><real_time_factor>true</real_time_factor><iterations>true</iterations>
      </plugin>
      <plugin filename="ImageDisplay" name="Drone camera">
        <gz-gui><title>AI detections (live)</title><property key="state" type="string">floating</property><property type="double" key="width">600</property><property type="double" key="height">500</property><property type="double" key="x">10</property><property type="double" key="y">10</property></gz-gui>
        <topic>/camo/image_annotated</topic><topic_picker>false</topic_picker>
      </plugin>
    </gui>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 200 0 0 0</pose>
      <diffuse>1.0 0.93 0.80 1</diffuse>
      <specular>0.35 0.33 0.3 1</specular>
      <attenuation><range>2000</range><constant>0.9</constant><linear>0.0</linear><quadratic>0.0</quadratic></attenuation>
      <direction>-0.5 0.25 -0.8</direction>
    </light>

    <include><name>terrain</name><uri>model://savanna_terrain</uri></include>
{body}
    <include><name>camo_drone</name><uri>model://camo_drone</uri><pose>-85 -85 {T.z(-85, -85) + 14.0:.2f} 0 0 0.78</pose></include>
  </world>
</sdf>
"""
    os.makedirs(os.path.join(WS, "worlds"), exist_ok=True)
    with open(os.path.join(WS, "worlds", "savanna.sdf"), "w") as f:
        f.write(sdf)
    with open(os.path.join(WS, "worlds", "targets.json"), "w") as f:
        json.dump(dict(seed=args.seed, terrain_size=A.TERRAIN_SIZE, targets=targets), f, indent=2)
    print(f"world written: {n} scenery objects, {len(targets)} targets")


if __name__ == "__main__":
    main()
