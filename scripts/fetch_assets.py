#!/usr/bin/env python3
"""Download CC0 assets from Poly Haven (https://polyhaven.com, CC0 licence) into assets_src/."""
import os, sys, json, urllib.request

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(WS, "assets_src")

TEXTURES = ["leafy_grass", "forest_leaves_04", "forest_ground_04", "brown_mud_leaves_01", "mossy_rock",
            "pine_bark", "bark_brown_02", "sparse_grass", "rocky_trail"]
MODELS = ["fern_02", "shrub_02", "shrub_03", "shrub_04", "rock_moss_set_01", "rock_moss_set_02",
          "tree_stump_01", "dead_tree_trunk", "grass_medium_01", "tree_small_02", "island_tree_02",
          "fir_sapling", "pine_sapling_small", "boulder_01"]


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "camo-sim-asset-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)
    print("  got", os.path.relpath(path, OUT), flush=True)


def files(asset):
    req = urllib.request.Request(f"https://api.polyhaven.com/files/{asset}", headers={"User-Agent": "camo-sim-asset-fetch"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "2k"
    for t in TEXTURES:
        f = files(t)
        for key, name in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough"), ("AO", "ao"), ("Displacement", "disp")):
            if key in f and res in f[key]:
                fmt = "jpg" if "jpg" in f[key][res] else "png"
                get(f[key][res][fmt]["url"], os.path.join(OUT, "textures", t, f"{name}.{fmt}"))
    for m in MODELS:
        g = files(m)["gltf"]["1k"]["gltf"]
        d = os.path.join(OUT, "models", m)
        get(g["url"], os.path.join(d, m + ".gltf"))
        for rel, info in g["include"].items():
            get(info["url"], os.path.join(d, rel))


if __name__ == "__main__":
    main()
