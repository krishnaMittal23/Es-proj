#!/usr/bin/env python3
"""Print node layout, bounds and materials of glTF files: inspect_gltf.py a.gltf b.gltf ..."""
import json, sys
for p in sys.argv[1:]:
    g = json.load(open(p))
    print("==", p.split("/")[-1], "nodes", len(g["nodes"]), "meshes", len(g["meshes"]))
    for n in g["nodes"][:10]:
        print("   ", n.get("name"), n.get("translation"), n.get("rotation"), n.get("scale"), "mesh", n.get("mesh"), "children", n.get("children"))
    mn, mx = [1e9] * 3, [-1e9] * 3
    for a in g["accessors"]:
        if "min" in a and len(a["min"]) == 3:
            mn = [min(x, y) for x, y in zip(mn, a["min"])]; mx = [max(x, y) for x, y in zip(mx, a["max"])]
    print("    bounds", [round(x, 2) for x in mn], [round(x, 2) for x in mx])
    for m in g["materials"]:
        print("    mat", m.get("name"), m.get("alphaMode"), m.get("doubleSided"))
