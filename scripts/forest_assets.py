#!/usr/bin/env python3
"""Forest asset builders: procedural trees/bushes with alpha-cut foliage cards, Poly Haven
glTF importer (Y-up -> Z-up, split into pieces), and camouflaged targets for the forest."""
import os, math, json, shutil, struct
import numpy as np
from meshlib import Mesh

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(WS, "assets_src")
PI = math.pi


def cell_uv(cell):
    u0, v0 = (cell % 2) * 0.5, (cell // 2) * 0.5
    return (u0 + 0.004, v0 + 0.004, u0 + 0.496, v0 + 0.496)


def foliage_normals(m, start, center, radii, upw=0.45):
    """Volumetric foliage lighting: normals point out of the crown (+ a bit up), not along the card."""
    c, r = np.asarray(center, float), np.asarray(radii, float)
    for i in range(start, len(m.v)):
        d = (np.asarray(m.v[i]) - c) / r
        d = d / (np.linalg.norm(d) + 1e-6) + np.array([0, 0, upw])
        m.vn[i] = tuple(d / np.linalg.norm(d))


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


# ----------------------------------------------------------------------------- trees
def broadleaf(seed, palette="summer", bark="bark_oak", H=None, crown_w=0.6, n_cards=330, cells=(0, 1, 2, 3), trunk=(0.2, 0.28)):
    rng = np.random.default_rng(seed)
    m = Mesh()
    H = H or rng.uniform(9.5, 15.0)
    th = H * rng.uniform(*trunk)
    r0 = 0.018 * H + rng.uniform(0, 0.06)
    p = np.array([0.0, 0.0, -0.5]); pts = [p]
    for k in range(4):
        p = p + np.array([rng.normal(0, 0.1), rng.normal(0, 0.1), (th + 0.5) / 4]); pts.append(p)
    for k in range(4):
        m.cylinder(bark, pts[k], pts[k + 1], r0 * (1 - 0.1 * k), r0 * (1 - 0.1 * (k + 1)), seg=9, cap=False)
    top = pts[-1]
    ch = H - th
    R = ch * crown_w * rng.uniform(0.85, 1.1)
    Rz = ch * 0.5
    cc = top + np.array([0, 0, Rz * 0.92])
    tips = []
    nb = int(rng.integers(5, 8))
    for i in range(nb):
        a = 2 * PI * i / nb + rng.uniform(-0.3, 0.3)
        el = rng.uniform(0.55, 1.05)
        L = R * rng.uniform(0.7, 0.95) / max(math.cos(el), 0.45) * 0.75
        s = top - np.array([0, 0, rng.uniform(0, th * 0.2)])
        d = np.array([math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)])
        e = s + d * L
        m.cylinder(bark, s, e, r0 * 0.5, 0.05, seg=6, cap=False)
        tips.append(e)
        for j in range(2):
            mid = s + d * L * rng.uniform(0.45, 0.8)
            d2 = d + rng.normal(0, 0.5, 3); d2[2] = abs(d2[2]) + 0.3; d2 = _unit(d2)
            e2 = mid + d2 * L * 0.45
            m.cylinder(bark, mid, e2, 0.07, 0.02, seg=5, cap=False)
            tips.append(e2)
    m.cylinder(bark, top, cc + np.array([0, 0, Rz * 0.55]), r0 * 0.6, 0.04, seg=6, cap=False)
    clumps = []
    for k in range(int(rng.integers(7, 11))):
        dv = rng.normal(0, 1, 3); dv[2] = abs(dv[2]) * 0.9 + 0.15; dv = _unit(dv)
        clumps.append(cc + dv * np.array([R, R, Rz]) * rng.uniform(0.5, 0.78))
    clumps += tips
    rad = np.array([R, R, Rz])
    start = len(m.v)
    for k in range(n_cards):
        p = clumps[k % len(clumps)] + rng.normal(0, 1, 3) * rad * 0.27
        q = np.linalg.norm((p - cc) / rad)
        if q > 1.05:
            p = cc + (p - cc) / q * 1.05
        s = rng.uniform(1.8, 2.8) * H / 13
        yaw, pitch = rng.uniform(0, 2 * PI), rng.uniform(0.25, 1.35)
        up = np.array([math.cos(yaw) * math.sin(pitch), math.sin(yaw) * math.sin(pitch), math.cos(pitch)])
        inner = np.linalg.norm((p - cc) / rad) < 0.6
        m.card(f"leaf_{palette}{'_in' if inner else ''}", p - up * s * 0.5, s * 0.9, s, yaw, pitch,
               cell_uv(cells[int(rng.integers(0, len(cells)))]))
    foliage_normals(m, start, cc, rad)
    return m, dict(H=H, R=R, kind="broadleaf")


def conifer(seed, H=None, base=None):
    rng = np.random.default_rng(seed)
    m = Mesh()
    H = H or rng.uniform(12.0, 18.0)
    base = base if base is not None else rng.uniform(0.3, 1.0) * H / 15
    Rmax = H * rng.uniform(0.16, 0.2)
    r0 = 0.02 * H
    m.cylinder("bark_pine", (0, 0, -0.4), (0, 0, H * 0.96), r0, 0.03, seg=8, cap=False)
    start = len(m.v)
    z = base
    while z < H - 0.5:
        rel = (z - base) / max(H - base, 1e-3)
        R = Rmax * (1 - rel) ** 0.92 + 0.3
        nbr = max(3, int(round(7 - 3 * rel)))
        a0 = rng.uniform(0, 2 * PI)
        for i in range(nbr):
            a = a0 + 2 * PI * i / nbr + rng.uniform(-0.25, 0.25)
            pitch = PI / 2 - 0.3 * rel + 0.22 * (1 - rel) + rng.normal(0, 0.07)
            L = R * rng.uniform(0.9, 1.15)
            w = max(0.45, L * 0.75)
            c = np.array([math.cos(a) * 0.08, math.sin(a) * 0.08, z])
            m.card("needle", c, w, L, a, pitch, cell_uv(int(rng.integers(0, 4))), bend=0.16)
            # near-vertical frond along the same branch -> silhouette from the side
            m.card("needle_in" if rel < 0.5 else "needle", c - np.array([0, 0, 0.1]), w * 0.85, L * 0.95, a + rng.uniform(-0.2, 0.2),
                   pitch + 0.05, cell_uv(int(rng.integers(0, 4))), bend=0.04, roll=rng.choice([-1, 1]) * rng.uniform(1.1, 1.45))
        z += rng.uniform(0.4, 0.55) * max(H / 15, 0.35)
    for yaw in (0.0, PI / 3, 2 * PI / 3):
        m.card("needle", (0, 0, H - 1.6), 0.8, 1.9, yaw, 0.0, cell_uv(int(rng.integers(0, 4))))
    foliage_normals(m, start, (0, 0, H * 0.42), (Rmax, Rmax, H * 0.6), upw=0.5)
    return m, dict(H=H, R=Rmax, kind="conifer")


def birch(seed):
    rng = np.random.default_rng(seed)
    return broadleaf(seed, palette="light", bark="bark_birch", H=rng.uniform(10, 14), crown_w=0.38, n_cards=220, trunk=(0.28, 0.36))


def fern(seed, size=None, n=None):
    """Procedural fern: arching alpha-cut fronds radiating from a crown (very cheap)."""
    rng = np.random.default_rng(seed)
    m = Mesh()
    size = size or rng.uniform(0.7, 1.2)
    n = n or int(rng.integers(8, 13))
    for k in range(n):
        yaw = 2 * PI * k / n + rng.uniform(-0.2, 0.2)
        m.card("fern", (0, 0, 0.02), size * 0.42, size * rng.uniform(0.8, 1.1), yaw, rng.uniform(0.55, 1.05),
               cell_uv(int(rng.integers(0, 4))), bend=0.35)
    foliage_normals(m, 0, (0, 0, -0.3 * size), (1, 1, 1), upw=0.8)
    return m


def snag(seed):
    """Dead standing tree: bare trunk and broken branches."""
    rng = np.random.default_rng(seed)
    m = Mesh()
    H = rng.uniform(5.5, 8.5)
    r0 = 0.25
    m.cylinder("bark_dead", (0, 0, -0.4), (rng.normal(0, .2), rng.normal(0, .2), H), r0, 0.06, seg=8, cap=False)
    for i in range(int(rng.integers(3, 6))):
        z = rng.uniform(H * 0.45, H * 0.9); a = rng.uniform(0, 2 * PI); L = rng.uniform(1, 2.8)
        m.cylinder("bark_dead", (0, 0, z), (math.cos(a) * L, math.sin(a) * L, z + L * rng.uniform(0.2, 0.8)), 0.08, 0.02, seg=5, cap=False)
    return m, dict(H=H, R=1.5, kind="snag")


def bush(seed, palette="summer", size=None, n=42):
    rng = np.random.default_rng(seed)
    m = Mesh()
    size = size or rng.uniform(1.2, 2.4)
    c = np.array([0, 0, size * 0.38])
    rad = np.array([size * 0.55, size * 0.55, size * 0.42])
    for k in range(n):
        d = rng.normal(0, 1, 3); d[2] = abs(d[2]) * 0.8 - 0.1; d = _unit(d)
        p = c + d * rad * rng.uniform(0.35, 1.0)
        s = size * rng.uniform(0.45, 0.65)
        yaw, pitch = rng.uniform(0, 2 * PI), rng.uniform(0.2, 1.3)
        up = np.array([math.cos(yaw) * math.sin(pitch), math.sin(yaw) * math.sin(pitch), math.cos(pitch)])
        inner = np.linalg.norm((p - c) / rad) < 0.55
        m.card(f"leaf_{palette}{'_in' if inner else ''}", p - up * s * 0.5, s * 0.9, s, yaw, pitch, cell_uv(int(rng.integers(0, 4))))
    foliage_normals(m, 0, c, rad, upw=0.6)
    return m


# ----------------------------------------------------------------------------- Poly Haven import
_CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_polyhaven(name, tex_dir, scale=1.0):
    """Return (pieces, mats): pieces = list of Mesh (each centred, base at z=0, Z-up);
    textures are copied into tex_dir and referenced as uri:tex/<file>."""
    d = os.path.join(SRC, "models", name)
    g = json.load(open(os.path.join(d, name + ".gltf")))
    buf = open(os.path.join(d, g["buffers"][0]["uri"]), "rb").read()

    def acc(i):
        a = g["accessors"][i]; bv = g["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = np.dtype(_CT[a["componentType"]]); nc = _NC[a["type"]]
        stride = bv.get("byteStride", 0) or dt.itemsize * nc
        raw = np.frombuffer(buf, np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * nc, offset=off)
        out = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * nc), (stride, 1)).copy()
        return out.view(dt).reshape(a["count"], nc)

    mname = "ph_" + name
    mat = g["materials"][0]
    os.makedirs(tex_dir, exist_ok=True)
    spec = dict(rough=0.85, double=bool(mat.get("doubleSided")), alpha=mat.get("alphaMode") == "MASK", cutoff=mat.get("alphaCutoff", 0.5))
    for key, slot in (("baseColorTexture", "tex"), ("normalTexture", "normal")):
        t = mat.get("pbrMetallicRoughness", {}).get(key) if key == "baseColorTexture" else mat.get(key)
        if t is not None:
            uri = g["images"][g["textures"][t["index"]]["source"]]["uri"]
            fn = f"{mname}_{slot}{os.path.splitext(uri)[1]}"
            shutil.copy(os.path.join(d, uri), os.path.join(tex_dir, fn))
            spec[slot] = "uri:tex/" + fn
    pieces = []
    for node in g["nodes"]:
        if "mesh" not in node:
            continue
        M = np.eye(3) * (node.get("scale", [1, 1, 1])[0])
        if "rotation" in node:
            M = _quat_mat(node["rotation"]) @ M
        for prim in g["meshes"][node["mesh"]]["primitives"]:
            P = acc(prim["attributes"]["POSITION"]).astype(float) @ M.T
            N = acc(prim["attributes"]["NORMAL"]).astype(float) @ M.T
            UV = acc(prim["attributes"]["TEXCOORD_0"]).astype(float)
            I = acc(prim["indices"]).reshape(-1, 3).astype(np.int64)
            P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1) * scale     # Y-up -> Z-up
            N = np.stack([N[:, 0], -N[:, 2], N[:, 1]], 1)
            P[:, 0] -= (P[:, 0].min() + P[:, 0].max()) / 2; P[:, 1] -= (P[:, 1].min() + P[:, 1].max()) / 2; P[:, 2] -= P[:, 2].min()
            UV = np.stack([UV[:, 0], 1.0 - UV[:, 1]], 1)               # glTF -> OBJ convention
            pm = Mesh(); pm.material(mname)
            pm.add_arrays(mname, P, UV, N, I)
            pm.extent = (P[:, 0].ptp(), P[:, 1].ptp(), P[:, 2].ptp())
            pieces.append(pm)
    return pieces, {mname: spec}


# ----------------------------------------------------------------------------- targets
def deer():
    m = Mesh()
    m.material("hide", (0.45, 0.32, 0.2)); m.material("belly", (0.75, 0.66, 0.55)); m.material("dark", (0.12, 0.09, 0.07))
    m.ellipsoid("hide", (0, 0, 1.0), (0.62, 0.24, 0.27), seg=14, rings=9)
    m.ellipsoid("belly", (0.0, 0, 0.88), (0.5, 0.2, 0.14), seg=10, rings=6)
    m.cylinder("hide", (0.5, 0, 1.1), (0.75, 0, 1.55), 0.12, 0.09, seg=8)
    m.ellipsoid("hide", (0.85, 0, 1.6), (0.2, 0.1, 0.1), seg=10, rings=7)
    m.ellipsoid("dark", (1.03, 0, 1.58), (0.04, 0.04, 0.04), seg=6, rings=4)
    for s in (-1, 1):
        m.ellipsoid("hide", (0.75, s * 0.09, 1.74), (0.04, 0.03, 0.09), seg=6, rings=4)
        m.cylinder("dark", (0.78, s * 0.05, 1.7), (0.7, s * 0.25, 2.05), 0.02, 0.01, seg=5)
        m.cylinder("dark", (0.73, s * 0.17, 1.88), (0.88, s * 0.25, 2.0), 0.015, 0.008, seg=5)
    for x in (-0.42, 0.42):
        for y in (-0.12, 0.12):
            m.cylinder("hide", (x, y, 0.95), (x + 0.02, y, 0.0), 0.05, 0.03, seg=6)
    m.ellipsoid("belly", (-0.62, 0, 1.12), (0.06, 0.06, 0.1), seg=6, rings=4)
    return m


def bear():
    m = Mesh()
    m.material("fur", (0.22, 0.15, 0.1)); m.material("dark", (0.06, 0.05, 0.04))
    m.ellipsoid("fur", (0, 0, 0.8), (0.85, 0.45, 0.45), seg=14, rings=9)
    m.ellipsoid("fur", (0.35, 0, 0.98), (0.4, 0.42, 0.38), seg=12, rings=8)
    m.ellipsoid("fur", (0.95, 0, 0.9), (0.25, 0.22, 0.22), seg=12, rings=8)
    m.ellipsoid("dark", (1.18, 0, 0.86), (0.09, 0.08, 0.07), seg=8, rings=5)
    for s in (-1, 1):
        m.ellipsoid("fur", (0.9, s * 0.15, 1.1), (0.06, 0.04, 0.06), seg=6, rings=4)
    for x in (-0.5, 0.5):
        for y in (-0.25, 0.25):
            m.cylinder("fur", (x, y, 0.65), (x, y, 0.0), 0.15, 0.13, seg=8)
    return m


def ghillie(humanoid_fn, seed=3):
    """Prone sniper in a ghillie suit: body covered with grass + leaf cards (strong camouflage)."""
    rng = np.random.default_rng(seed)
    m = humanoid_fn("prone", "camo_forest_b.png")
    pts = np.asarray(m.v)
    zmax = pts[:, 2].max()
    for k in range(70):
        p = pts[int(rng.integers(0, len(pts)))].copy()
        p[2] = max(0.0, p[2] - 0.05)
        s = rng.uniform(0.35, 0.6)
        yaw, pitch = rng.uniform(0, 2 * PI), rng.uniform(0.2, 1.0)
        if rng.random() < 0.6:
            from forest_textures import GRASS_UV
            m.card("g_grass", p, s, s * 0.8, yaw, pitch, GRASS_UV[int(rng.integers(0, len(GRASS_UV)))])
        else:
            m.card("g_leaf", p, s, s, yaw, pitch, cell_uv(int(rng.integers(0, 4))))
    return m


def camo_net_jeep(jeep_fn, tex, seed=4):
    """Vehicle under camouflage netting with leaf cards scattered on top."""
    rng = np.random.default_rng(seed)
    m = jeep_fn(tex)
    for k in range(60):
        x, y = rng.uniform(-2.3, 2.3), rng.uniform(-1.3, 1.3)
        z = 1.85 - 0.12 * (abs(y) / 1.3) ** 2 - (0.6 if abs(x) > 1.6 else 0)
        s = rng.uniform(0.6, 1.0)
        yaw, pitch = rng.uniform(0, 2 * PI), rng.uniform(1.1, 1.5)
        m.card("g_leaf", (x, y, z - 0.1), s, s, yaw, pitch, cell_uv(int(rng.integers(0, 4))))
    return m
