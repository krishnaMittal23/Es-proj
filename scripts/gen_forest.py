#!/usr/bin/env python3
"""Generate the photo-realistic forest world.

usage: gen_forest.py [--seed N] [--targets K] [--force-textures]
writes models/forest_terrain, models/forest_veg (tiled vegetation), models/forest_far (horizon),
target models, worlds/forest.sdf, worlds/targets.json, worlds/world_info.json
"""
import os, json, math, argparse, shutil, time
import numpy as np
from PIL import Image
from scipy import ndimage
from meshlib import Mesh
import gen_assets as GA
import forest_textures as FT
import forest_assets as FA

WS, MODELS, SRC = GA.WS, GA.MODELS, FT.SRC
SIZE, HEIGHT, N = 240.0, 18.0, 513
HALF = SIZE / 2
TILE = 30.0
NT = int(SIZE / TILE)
VEG_M = os.path.join(MODELS, "forest_veg", "meshes"); VEG_T = os.path.join(VEG_M, "tex")
TER = os.path.join(MODELS, "forest_terrain")
FAR_M = os.path.join(MODELS, "forest_far", "meshes")
BASE_XY = (-85.0, -85.0)

CLEARINGS = [(-85, -85, 17), (-40, 30, 17), (30, -45, 19), (55, 55, 15), (-60, -20, 12), (10, 15, 13), (-15, 70, 12),
             (75, -5, 12), (-80, 45, 12), (40, 90, 10), (-95, 95, 10), (95, -80, 12)]
TRAIL = [(-130, -100), (-85, -85), (-55, -62), (-30, -48), (-5, -22), (15, 5), (40, 15), (62, 38), (90, 48), (130, 72)]


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


def catmull(pts, step=0.5):
    P = np.asarray(pts, float); P = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def stream_y(x):
    return -28 + 22 * np.sin(x / SIZE * 2 * np.pi + 0.6) + 6 * np.sin(x / SIZE * 2 * np.pi * 2.7)


# ----------------------------------------------------------------------------- terrain
class ForestTerrain:
    def __init__(self):
        n = N
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
        X, Y = (xx - 0.5) * SIZE, (0.5 - yy) * SIZE
        self.X, self.Y = X, Y
        base = GA.fbm(n, 7, seed=3, persistence=0.5)
        g = lambda cx, cy, s: np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
        dy = Y - stream_y(X)
        valley, stream = np.exp(-dy ** 2 / (2 * 16 ** 2)), np.exp(-dy ** 2 / (2 * 3.2 ** 2))
        h = 0.32 * base + 0.55 * g(70, 62, 38) + 0.35 * g(-88, 40, 30) + 0.3 * g(95, -85, 30) - 0.18 * valley
        h = (h - h.min()) / (h.max() - h.min())
        h = h - 0.075 * stream
        self.h = (h - h.min()) / (h.max() - h.min())
        self.stream_d = np.abs(dy)
        # trail distance field
        tp = catmull(TRAIL, 0.4)
        grid = np.ones((n, n), bool)
        cols = np.clip(np.round((tp[:, 0] / SIZE + 0.5) * (n - 1)).astype(int), 0, n - 1)
        rows = np.clip(np.round((0.5 - tp[:, 1] / SIZE) * (n - 1)).astype(int), 0, n - 1)
        grid[rows, cols] = False
        self.trail_d = ndimage.distance_transform_edt(grid) * SIZE / (n - 1)
        f = GA.fbm(n, 5, seed=21)
        for cx, cy, r in CLEARINGS:
            f = f - 0.9 * g(cx, cy, r / 1.41)
        forest = smoothstep(0.30, 0.48, f)
        forest *= 1 - 0.95 * np.exp(-(self.trail_d / 3.5) ** 2)
        forest *= 1 - 0.85 * np.exp(-dy ** 2 / (2 * 4.5 ** 2))
        forest = np.maximum(forest, smoothstep(HALF - 14, HALF - 3, np.maximum(np.abs(X), np.abs(Y))))
        self.forest = forest
        self.moss = GA.fbm(n, 5, seed=33)

    def sample(self, arr, x, y):
        n = N
        u = min(max((x / SIZE + 0.5) * (n - 1), 0), n - 1.001)
        v = min(max((0.5 - y / SIZE) * (n - 1), 0), n - 1.001)
        i, j = int(v), int(u); fu, fv = u - j, v - i
        return float(arr[i, j] * (1 - fu) * (1 - fv) + arr[i, j + 1] * fu * (1 - fv) + arr[i + 1, j] * (1 - fu) * fv + arr[i + 1, j + 1] * fu * fv)

    def z(self, x, y):
        return self.sample(self.h, x, y) * HEIGHT

    def slope(self, x, y, e=1.0):
        return max(abs(self.z(x + e, y) - self.z(x - e, y)), abs(self.z(x, y + e) - self.z(x, y - e))) / (2 * e)


def skirt_z(T, x, y):
    e = max(abs(x), abs(y)) - HALF
    edge = T.z(min(max(x, -HALF), HALF), min(max(y, -HALF), HALF))
    if e <= 0:
        return edge
    hills = 10 + 9 * math.sin(x / 61 + 1.3) * math.cos(y / 83 - 0.4) + 6 * math.sin(x / 29 + y / 41) + 0.04 * e
    w = smoothstep(0, 90, e)
    return float((1 - w) * edge + w * hills)


# ----------------------------------------------------------------------------- textures
def build_textures(force=False):
    os.makedirs(VEG_T, exist_ok=True)
    jobs = [
        ("leaf_summer.png", lambda p: FT.broadleaf_atlas(p, [(62, 98, 44), (78, 118, 52), (96, 134, 58), (56, 88, 40), (104, 140, 64)], 1)),
        ("leaf_dark.png", lambda p: FT.broadleaf_atlas(p, [(44, 72, 36), (56, 86, 42), (66, 98, 48), (50, 80, 46)], 2, leaf_len=(40, 64))),
        ("leaf_light.png", lambda p: FT.broadleaf_atlas(p, [(112, 150, 62), (128, 164, 72), (98, 140, 56), (140, 170, 80)], 3, leaf_len=(30, 48), density=1.25)),
        ("needle.png", lambda p: FT.conifer_atlas(p, [(34, 62, 42), (42, 74, 46), (52, 84, 50), (30, 54, 40)], 4)),
        ("grass.png", lambda p: FT.grass_atlas(p)),
        ("fern.png", lambda p: FT.fern_atlas(p)),
        ("bark_birch.jpg", lambda p: FT.birch_bark(p)),
        ("far_tree.png", lambda p: FT.far_tree_atlas(p)),
        ("far_canopy.jpg", lambda p: FT.canopy_texture(p)),
    ]
    for fn, fnc in jobs:
        p = os.path.join(VEG_T, fn)
        if force or not os.path.exists(p):
            fnc(p); print("  texture", fn, flush=True)
    for src, dst in (("pine_bark", "bark_pine"), ("bark_brown_02", "bark_oak")):
        if force or not os.path.exists(os.path.join(VEG_T, dst + ".jpg")):
            FT.resized(os.path.join(SRC, "textures", src, "diff.jpg"), os.path.join(VEG_T, dst + ".jpg"), 1024)
            FT.resized(os.path.join(SRC, "textures", src, "nor.jpg"), os.path.join(VEG_T, dst + "_nor.jpg"), 1024)
    camo = {
        "camo_forest_a.png": ([(0.0, (46, 58, 34)), (0.35, (78, 92, 50)), (0.55, (98, 82, 56)), (0.8, (36, 34, 26)), (1.0, (88, 104, 60))], 41),
        "camo_forest_b.png": ([(0.0, (60, 72, 40)), (0.3, (92, 100, 58)), (0.6, (70, 58, 40)), (0.85, (44, 52, 32)), (1.0, (110, 116, 72))], 42),
        "camo_forest_c.png": ([(0.0, (52, 64, 36)), (0.4, (40, 46, 30)), (0.7, (86, 96, 54)), (1.0, (74, 62, 44))], 43),
    }
    for fn, (stops, seed) in camo.items():
        p = os.path.join(VEG_T, fn)
        if force or not os.path.exists(p):
            FT.camo_texture(p, stops, seed)
    for fn, col in (("plain_red.png", (205, 50, 38)), ("plain_blue.png", (40, 85, 195))):
        a = np.zeros((64, 64, 3), np.uint8); a[...] = col
        Image.fromarray(a).save(os.path.join(VEG_T, fn))


def leaf_mats():
    M = {}
    for pal in ("summer", "dark", "light"):
        M[f"leaf_{pal}"] = dict(tex=f"uri:tex/leaf_{pal}.png", alpha=True, double=True, rough=0.75, cutoff=0.45)
        M[f"leaf_{pal}_in"] = dict(tex=f"uri:tex/leaf_{pal}.png", alpha=True, double=True, rough=0.85, cutoff=0.45, color=(0.58, 0.6, 0.55))
    M["needle"] = dict(tex="uri:tex/needle.png", alpha=True, double=True, rough=0.8, cutoff=0.35)
    M["needle_in"] = dict(tex="uri:tex/needle.png", alpha=True, double=True, rough=0.9, cutoff=0.35, color=(0.6, 0.62, 0.6))
    M["bark_oak"] = dict(tex="uri:tex/bark_oak.jpg", normal="uri:tex/bark_oak_nor.jpg", rough=0.95)
    M["bark_dead"] = dict(tex="uri:tex/bark_oak.jpg", normal="uri:tex/bark_oak_nor.jpg", rough=0.95, color=(0.7, 0.68, 0.66))
    M["bark_pine"] = dict(tex="uri:tex/bark_pine.jpg", normal="uri:tex/bark_pine_nor.jpg", rough=0.95)
    M["bark_birch"] = dict(tex="uri:tex/bark_birch.jpg", rough=0.8)
    M["grass"] = dict(tex="uri:tex/grass.png", alpha=True, double=True, rough=0.9, cutoff=0.4)
    M["fern"] = dict(tex="uri:tex/fern.png", alpha=True, double=True, rough=0.8, cutoff=0.45)
    M["far_tree"] = dict(tex="uri:tex/far_tree.png", alpha=True, double=True, rough=1.0, cutoff=0.4)
    M["far_canopy"] = dict(tex="uri:tex/far_canopy.jpg", rough=1.0)
    return M


# ----------------------------------------------------------------------------- terrain bake + mesh
def bake_terrain(T, trees, R=4096):
    L = {}
    # (layer, photo texture, tile metres, colour-grade tint, grade amount, brightness)
    for key, name, tile, tint, amt, br in (
            ("grass", "leafy_grass", 3.2, (0.6, 1.0, 0.34), 0.8, 0.92), ("grass2", "sparse_grass", 3.6, (0.62, 1.0, 0.3), 0.55, 1.25),
            ("leaves", "forest_leaves_04", 3.0, (1, 1, 1), 0.0, 0.78), ("moss", "forest_ground_04", 3.5, (0.55, 0.8, 0.32), 0.65, 0.95),
            ("mud", "brown_mud_leaves_01", 3.0, (1, 1, 1), 0.0, 0.95), ("trail", "rocky_trail", 3.4, (1, 1, 1), 0.0, 0.92)):
        d = np.asarray(Image.open(os.path.join(SRC, "textures", name, "diff.jpg")).convert("RGB"), np.float32)
        if amt:
            lum = d.mean(-1, keepdims=True)
            t = np.array(tint, np.float32); t = t / t.mean()
            d = (1 - amt) * d + amt * lum * t
        d = np.clip(d * br, 0, 255).astype(np.uint8)
        nn = np.asarray(Image.open(os.path.join(SRC, "textures", name, "nor.jpg")).convert("RGB"))
        L[key] = (d, nn, tile)
    up = lambda a: np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((R, R), Image.BILINEAR))
    forest = up(T.forest)
    trail = up(np.exp(-(T.trail_d / 1.25) ** 2) + 0.35 * np.exp(-(T.trail_d / 2.2) ** 2))
    trail = np.clip(trail, 0, 1)
    bank = up(np.exp(-(T.stream_d / 4.2) ** 2))
    moss_n = up(smoothstep(0.35, 0.65, T.moss))
    gvar = up(smoothstep(0.4, 0.6, GA.fbm(N, 5, seed=55)))
    macro = up(0.86 + 0.28 * GA.fbm(N, 6, seed=56))
    # canopy shadow / ambient occlusion from trees
    A = 1024; ao = np.ones((A, A), np.float32)
    px = A / SIZE
    for t in trees:
        if max(abs(t["x"]), abs(t["y"])) > HALF:
            continue
        r = t["R"] * t["s"] * 1.15
        cx, cy = (t["x"] + HALF) * px, (HALF - t["y"]) * px
        rr = int(r * px * 1.8) + 2
        i0, i1 = max(0, int(cy) - rr), min(A, int(cy) + rr); j0, j1 = max(0, int(cx) - rr), min(A, int(cx) + rr)
        if i0 >= i1 or j0 >= j1:
            continue
        ii, jj = np.mgrid[i0:i1, j0:j1]
        d2 = ((ii - cy) ** 2 + (jj - cx) ** 2) / (r * px) ** 2
        ao[i0:i1, j0:j1] *= 1 - 0.38 * np.exp(-d2 * 1.4) - 0.25 * np.exp(-d2 * 25)
    ao = np.asarray(Image.fromarray(ao, "F").resize((R, R), Image.BILINEAR))
    out_d = np.zeros((R, R, 3), np.uint8); out_n = np.zeros((R, R, 3), np.uint8)
    xs = ((np.arange(R) + 0.5) / R - 0.5) * SIZE
    ys = (0.5 - (np.arange(R) + 0.5) / R) * SIZE

    def gather(key, rows, cols_x, scale=1.0, off=0.0):
        d, nn, tile = L[key]; S = d.shape[0]; t = tile * scale
        ui = (((cols_x + off) / t) * S).astype(np.int64) % S
        vi = (((-ys[rows] + off) / t) * S).astype(np.int64) % S
        return d[vi[:, None], ui[None, :]].astype(np.float32), nn[vi[:, None], ui[None, :]].astype(np.float32)

    CH = 256
    for r0 in range(0, R, CH):
        rows = np.arange(r0, min(R, r0 + CH)); sl = slice(r0, r0 + len(rows))
        f, tr, bk, mo, gv = forest[sl], trail[sl], bank[sl], moss_n[sl], gvar[sl]
        w_tr = tr
        w_bk = bk * (1 - w_tr)
        rest = np.clip(1 - w_tr - w_bk, 0, 1)
        w = {"trail": w_tr, "mud": w_bk, "grass": rest * (1 - f) * (1 - gv * 0.6), "grass2": rest * (1 - f) * gv * 0.6,
             "leaves": rest * f * mo, "moss": rest * f * (1 - mo)}
        acc_d = np.zeros((len(rows), R, 3), np.float32); acc_n = np.zeros_like(acc_d); tot = np.zeros((len(rows), R, 1), np.float32)
        for key, wk in w.items():
            d1, n1 = gather(key, rows, xs)
            d2, n2 = gather(key, rows, xs, 2.37, 17.3)
            wk = wk[..., None]
            acc_d += wk * (0.62 * d1 + 0.38 * d2); acc_n += wk * (0.62 * n1 + 0.38 * n2); tot += wk
        acc_d /= np.maximum(tot, 1e-4); acc_n /= np.maximum(tot, 1e-4)
        shade = (macro[sl] * ao[sl])[..., None]
        acc_d *= shade * (1 - 0.18 * f[..., None])
        out_d[sl] = np.clip(acc_d, 0, 255).astype(np.uint8)
        nv = acc_n / 127.5 - 1; nv /= np.linalg.norm(nv, axis=-1, keepdims=True) + 1e-6
        out_n[sl] = ((nv + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return out_d, out_n


def write_terrain(T, trees):
    os.makedirs(os.path.join(TER, "meshes", "tex"), exist_ok=True); os.makedirs(os.path.join(TER, "textures"), exist_ok=True)
    Image.fromarray((T.h * 255).round().astype(np.uint8)).save(os.path.join(TER, "textures", "heightmap.png"))
    np.save(os.path.join(TER, "textures", "heightmap.npy"), T.h.astype(np.float32))
    t0 = time.time()
    d, nrm = bake_terrain(T, trees)
    Image.fromarray(d).save(os.path.join(TER, "meshes", "tex", "terrain_diff.jpg"), quality=90)
    Image.fromarray(nrm).save(os.path.join(TER, "meshes", "tex", "terrain_nor.jpg"), quality=92)
    print(f"  terrain bake {time.time() - t0:.0f}s", flush=True)
    n = N; dx = SIZE / (n - 1)
    Hm = T.h * HEIGHT
    P = np.stack([T.X.ravel(), T.Y.ravel(), Hm.ravel()], 1)
    gx = np.gradient(Hm, axis=1) / dx; gy = -np.gradient(Hm, axis=0) / dx
    Nn = np.stack([-gx, -gy, np.ones_like(gx)], -1).reshape(-1, 3); Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    UV = np.stack([(jj / (n - 1)).ravel(), (1 - ii / (n - 1)).ravel()], 1)
    a = (ii[:-1, :-1] * n + jj[:-1, :-1]).ravel()
    tris = np.concatenate([np.stack([a, a + n, a + 1], 1), np.stack([a + 1, a + n, a + n + 1], 1)])
    m = Mesh(); m.material("terrain")
    m.add_arrays("terrain", P, UV, Nn, tris)
    m.write_glb(os.path.join(TER, "meshes", "terrain.glb"),
                {"terrain": dict(tex="uri:tex/terrain_diff.jpg", normal="uri:tex/terrain_nor.jpg", rough=0.97)})
    GA.write_model(TER, "forest_terrain", f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="forest_terrain">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry><heightmap>
        <uri>model://forest_terrain/textures/heightmap.png</uri><size>{SIZE} {SIZE} {HEIGHT}</size><pos>0 0 0</pos>
      </heightmap></geometry></collision>
      <visual name="visual"><geometry><mesh><uri>model://forest_terrain/meshes/terrain.glb</uri></mesh></geometry></visual>
    </link>
  </model>
</sdf>""")


# ----------------------------------------------------------------------------- horizon
def write_far(T, rng):
    os.makedirs(FAR_M, exist_ok=True)
    if os.path.islink(os.path.join(FAR_M, "tex")) or os.path.exists(os.path.join(FAR_M, "tex")):
        shutil.rmtree(os.path.join(FAR_M, "tex"))
    os.makedirs(os.path.join(FAR_M, "tex"))
    for fn in ("far_tree.png", "far_canopy.jpg"):
        shutil.copy(os.path.join(VEG_T, fn), os.path.join(FAR_M, "tex", fn))
    st, E = 10.0, 600.0
    xs = np.arange(-E, E + 0.1, st); k = len(xs)
    P, UV = [], []
    for y in xs[::-1]:
        for x in xs:
            inside = max(abs(x), abs(y)) < HALF
            z = skirt_z(T, x, y) - (0.6 if inside else (0.15 if max(abs(x), abs(y)) == HALF else 0))
            P.append((x, y, z)); UV.append((x / 34.0, y / 34.0))
    P = np.array(P); tris = []
    for i in range(k - 1):
        for j in range(k - 1):
            cx, cy = (xs[j] + xs[j + 1]) / 2, (xs[::-1][i] + xs[::-1][i + 1]) / 2
            if max(abs(cx), abs(cy)) < HALF - st:
                continue
            a = i * k + j
            tris += [(a, a + k, a + 1), (a + 1, a + k, a + k + 1)]
    m = Mesh(); m.material("far_canopy")
    m.add_arrays("far_canopy", P, UV, np.tile([0, 0, 1.0], (len(P), 1)), np.array(tris))
    # impostor trees beyond the detailed rim
    cnt = 0
    for _ in range(26000):
        x, y = rng.uniform(-E * 0.92, E * 0.92, 2)
        e = max(abs(x), abs(y))
        if e < HALF + 16:
            continue
        s = rng.uniform(11, 19)
        cell = int(rng.integers(0, 4))
        z = skirt_z(T, x, y) - 0.5
        yaw = rng.uniform(0, math.pi)
        for dyaw in (0, math.pi / 2):
            m.card("far_tree", (x, y, z), s * 0.75, s, yaw + dyaw, 0.0, FA.cell_uv(cell), normal_up=0.5)
        cnt += 1
    m.write_glb(os.path.join(FAR_M, "far.glb"), {"far_canopy": leaf_mats()["far_canopy"], "far_tree": leaf_mats()["far_tree"]})
    GA.write_model(os.path.join(MODELS, "forest_far"), "forest_far", """<?xml version="1.0"?>
<sdf version="1.9"><model name="forest_far"><static>true</static><link name="link">
  <visual name="v"><cast_shadows>false</cast_shadows><geometry><mesh><uri>model://forest_far/meshes/far.glb</uri></mesh></geometry></visual>
</link></model></sdf>""")
    print(f"  horizon: {cnt} impostor trees", flush=True)


# ----------------------------------------------------------------------------- targets
TARGET_TYPES = [
    ("person_camo_stand", 1.2, True), ("person_camo_crouch", 1.2, True), ("person_ghillie_prone", 1.4, True),
    ("person_plain_stand", 1.2, False), ("person_plain_prone", 1.2, False),
    ("tent_camo", 2.6, True), ("jeep_camo_net", 3.2, True), ("deer", 1.4, False), ("bear", 1.6, False),
]


def build_targets():
    d = os.path.join(VEG_T)
    tex = lambda f: os.path.join(d, f)
    leaf = dict(tex=tex("leaf_dark.png"), alpha=True, double=True, cutoff=0.45, rough=0.8)
    grass = dict(tex=tex("grass.png"), alpha=True, double=True, cutoff=0.4, rough=0.9)
    skin = dict(color=(0.55, 0.4, 0.3), rough=0.7); dark = dict(color=(0.08, 0.08, 0.08), rough=0.6)
    specs = {
        "person_camo_stand": (GA.humanoid("stand", ""), {"cloth": dict(tex=tex("camo_forest_a.png")), "skin": skin, "dark": dark}, 1),
        "person_camo_crouch": (GA.humanoid("crouch", ""), {"cloth": dict(tex=tex("camo_forest_c.png")), "skin": skin, "dark": dark}, 1),
        "person_ghillie_prone": (FA.ghillie(GA.humanoid), {"cloth": dict(tex=tex("camo_forest_b.png")), "skin": skin, "dark": dark, "g_grass": grass, "g_leaf": leaf}, 1),
        "person_plain_stand": (GA.humanoid("stand", ""), {"cloth": dict(tex=tex("plain_red.png")), "skin": skin, "dark": dark}, 1),
        "person_plain_prone": (GA.humanoid("prone", ""), {"cloth": dict(tex=tex("plain_blue.png")), "skin": skin, "dark": dark}, 1),
        "tent_camo": (GA.tent(""), {"cloth": dict(tex=tex("camo_forest_a.png"), double=True)}, 2),
        "jeep_camo_net": (FA.camo_net_jeep(GA.jeep, ""), {"body": dict(tex=tex("camo_forest_b.png"), rough=0.6), "dark": dark,
                                                         "glass": dict(color=(0.1, 0.13, 0.15), rough=0.1), "g_leaf": leaf}, 2),
        "deer": (FA.deer(), {"hide": dict(color=(0.45, 0.31, 0.19)), "belly": dict(color=(0.72, 0.62, 0.5)), "dark": dark}, 3),
        "bear": (FA.bear(), {"fur": dict(color=(0.2, 0.14, 0.09), rough=0.95), "dark": dark}, 3),
    }
    for name, (mesh, mats, label) in specs.items():
        md = os.path.join(MODELS, name)
        mesh.write_glb(os.path.join(md, "meshes", name + ".glb"), mats)
        GA.write_model(md, name, GA.static_mesh_model(name, name + ".glb", None, label))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", type=int, default=9)
    ap.add_argument("--force-textures", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    build_textures(args.force_textures)
    T = ForestTerrain()
    MATS = leaf_mats()

    # ---------------- tree variants
    V = {"broad_summer": [FA.broadleaf(100 + i, "summer") for i in range(4)],
         "broad_dark": [FA.broadleaf(200 + i, "dark", cells=(0, 1, 2, 3)) for i in range(3)],
         "birch": [FA.birch(300 + i) for i in range(3)],
         "conifer": [FA.conifer(400 + i) for i in range(6)],
         "young_conifer": [FA.conifer(500 + i, H=np.random.default_rng(500 + i).uniform(3, 6.5), base=0.15) for i in range(3)],
         "young_broad": [FA.broadleaf(600 + i, "summer", H=np.random.default_rng(600 + i).uniform(4, 7), n_cards=90) for i in range(2)],
         "snag": [FA.snag(700 + i) for i in range(2)]}
    print(f"  tree variants built {time.time() - t0:.0f}s", flush=True)

    # ---------------- tree placement
    trees = []
    sp = 5.2
    for gx in np.arange(-HALF - 18, HALF + 18, sp):
        for gy in np.arange(-HALF - 18, HALF + 18, sp):
            x, y = gx + rng.uniform(-2.5, 2.5), gy + rng.uniform(-2.5, 2.5)
            e = max(abs(x), abs(y))
            if e < HALF - 1:
                f = T.sample(T.forest, x, y)
                if math.hypot(x - BASE_XY[0], y - BASE_XY[1]) < 14:
                    continue
                if T.sample(T.trail_d, x, y) < 3.0 or T.sample(T.stream_d, x, y) < 4.5:
                    continue
                if rng.random() > f ** 1.3 and rng.random() > 0.02:
                    continue
                z = T.z(x, y)
            elif e < HALF + 18:
                if rng.random() > 0.85:
                    continue
                f, z = 1.0, skirt_z(T, x, y)
            else:
                continue
            elev = z / HEIGHT
            pc = float(np.clip(0.18 + 0.7 * elev + 0.35 * (T.sample(T.moss, x, y) - 0.5), 0.08, 0.9))
            r = rng.random()
            if r < 0.01:
                kind = "snag"
            elif r < 0.01 + pc * 0.95:
                kind = "young_conifer" if rng.random() < 0.12 else "conifer"
            else:
                kind = "birch" if (f < 0.8 and rng.random() < 0.3) else ("young_broad" if rng.random() < 0.1 else
                                                                        ("broad_dark" if rng.random() < 0.4 else "broad_summer"))
            vi = int(rng.integers(0, len(V[kind])))
            info = V[kind][vi][1]
            trees.append(dict(x=x, y=y, z=z - 0.25, kind=kind, vi=vi, yaw=rng.uniform(0, 2 * math.pi),
                              s=rng.uniform(0.85, 1.08 if kind == "conifer" else 1.12), R=info["R"], H=info["H"]))
    print(f"  {len(trees)} trees placed", flush=True)
    tree_xy = np.array([(t["x"], t["y"]) for t in trees])

    # ---------------- targets (visible from the air: clearings / forest edges)
    targets = []
    placed = []
    for i in range(args.targets):
        name, rad, camo = TARGET_TYPES[i % len(TARGET_TYPES)]
        for _ in range(4000):
            x, y = rng.uniform(-70, 70, 2)
            f = T.sample(T.forest, x, y)
            if not (0.05 < f < 0.6) and rng.random() > 0.15:
                continue
            if T.sample(T.trail_d, x, y) < 3 or T.sample(T.stream_d, x, y) < 6 or T.slope(x, y) > 0.25:
                continue
            if math.hypot(x - BASE_XY[0], y - BASE_XY[1]) < 25:
                continue
            if len(tree_xy) and np.min(np.hypot(tree_xy[:, 0] - x, tree_xy[:, 1] - y)) < 3.5 + rad:
                continue
            if any(math.hypot(x - px, y - py) < 16 for px, py in placed):
                continue
            break
        else:
            continue
        placed.append((x, y))
        z = T.z(x, y) - 0.02
        targets.append(dict(id=i, model=name, cls=name.split("_")[0], x=round(x, 2), y=round(y, 2), z=round(z, 2),
                            yaw=float(rng.uniform(0, 2 * math.pi)), camouflaged=camo, r=rad))
    print(f"  {len(targets)} targets placed", flush=True)

    def near_target(x, y, r):
        return any(math.hypot(x - t["x"], y - t["y"]) < r + t["r"] for t in targets)

    # ---------------- vegetation tiles
    tiles = {}

    def tile(x, y, layer):
        i = int(np.clip((x + HALF) // TILE, 0, NT - 1)); j = int(np.clip((y + HALF) // TILE, 0, NT - 1))
        k = (i, j, layer)
        if k not in tiles:
            tiles[k] = Mesh()
        return tiles[k]

    for t in trees:
        tile(t["x"], t["y"], "trees").merge(V[t["kind"]][t["vi"]][0], (t["x"], t["y"], t["z"]), t["yaw"], t["s"])
    print(f"  trees merged {time.time() - t0:.0f}s", flush=True)

    # grass + flowers
    ng = 0
    fern_v = [FA.fern(900 + i) for i in range(6)]
    for gx in np.arange(-HALF + 0.4, HALF - 0.4, 0.7):
        for gy in np.arange(-HALF + 0.4, HALF - 0.4, 0.7):
            x, y = gx + rng.uniform(-0.35, 0.35), gy + rng.uniform(-0.35, 0.35)
            f = T.sample(T.forest, x, y)
            td, sd = T.sample(T.trail_d, x, y), T.sample(T.stream_d, x, y)
            meadow = (1 - f) * (1 - math.exp(-(td / 1.1) ** 2)) * (1 - math.exp(-(sd / 2.2) ** 2))
            if near_target(x, y, 0.4) or math.hypot(x - BASE_XY[0], y - BASE_XY[1]) < 3.5:
                continue
            m = tile(x, y, "under")
            if rng.random() < meadow * 0.85 + 0.04:
                flower = rng.random() < 0.05 * meadow
                photo = rng.random() < 0.3
                uv = FT.FLOWER_UV if flower else FT.GRASS_UV[int(rng.integers(0, 5)) if photo else int(rng.integers(5, 8))]
                s = rng.uniform(0.38, 0.7) * (0.8 + 0.4 * meadow)
                z = T.z(x, y) - 0.05
                yaw = rng.uniform(0, math.pi)
                for dyaw in (0, math.pi / 3, 2 * math.pi / 3):
                    m.card("grass", (x, y, z), s * 1.2, s, yaw + dyaw, rng.normal(0, 0.1), uv, normal_up=0.7)
                ng += 1
            elif f > 0.35 and rng.random() < 0.075 * f and td > 1.8 and sd > 2.5:
                m.merge(fern_v[int(rng.integers(0, 6))], (x, y, T.z(x, y) - 0.03), rng.uniform(0, 2 * math.pi), rng.uniform(0.7, 1.35))
    print(f"  {ng} grass clumps {time.time() - t0:.0f}s", flush=True)

    # undergrowth: Poly Haven ferns/plants, procedural bushes, rocks, stumps, logs
    ph = {}
    for nm, sc in (("fern_02", 2.3), ("shrub_03", 3.0), ("rock_moss_set_01", 1.0), ("rock_moss_set_02", 1.0),
                   ("tree_stump_01", 1.1), ("dead_tree_trunk", 2.0), ("boulder_01", 1.6)):
        pieces, mats = FA.load_polyhaven(nm, VEG_T, sc)
        ph[nm] = pieces; MATS.update(mats)
    bushes = {pal: [FA.bush(800 + i + 10 * k, pal) for i in range(4)] for k, pal in enumerate(("summer", "dark", "light"))}

    def scatter(count, pred, fn, tries=40000):
        c = 0
        for _ in range(tries):
            if c >= count:
                break
            x, y = rng.uniform(-HALF + 2, HALF - 2, 2)
            if not pred(x, y):
                continue
            fn(x, y); c += 1
        return c

    f_at = lambda x, y: T.sample(T.forest, x, y)
    edge = lambda x, y: 0.15 < f_at(x, y) < 0.8
    clear_of = lambda x, y, r=1.6: (not near_target(x, y, r)) and T.sample(T.trail_d, x, y) > 1.8 and T.sample(T.stream_d, x, y) > 2.5 \
        and math.hypot(x - BASE_XY[0], y - BASE_XY[1]) > 6

    def put_ph(nm, layer, s=(0.8, 1.25), sink=0.05, yaw=None):
        def f(x, y):
            p = ph[nm][int(rng.integers(0, len(ph[nm])))]
            tile(x, y, layer).merge(p, (x, y, T.z(x, y) - sink), rng.uniform(0, 2 * math.pi) if yaw is None else yaw, rng.uniform(*s))
        return f

    n_f = scatter(160,lambda x, y: (edge(x, y) or 3 < T.sample(T.stream_d, x, y) < 9 or 2.2 < T.sample(T.trail_d, x, y) < 5) and clear_of(x, y),
                  put_ph("fern_02", "under"))
    n_s = scatter(70, lambda x, y: edge(x, y) and clear_of(x, y), put_ph("shrub_03", "under"))

    def put_bush(x, y):
        pal = ("summer", "dark", "light")[int(rng.integers(0, 3))]
        tile(x, y, "under").merge(bushes[pal][int(rng.integers(0, 4))], (x, y, T.z(x, y) - 0.1), rng.uniform(0, 2 * math.pi), rng.uniform(0.7, 1.3))
    n_b = scatter(420, lambda x, y: (edge(x, y) or rng.random() < 0.12) and clear_of(x, y, 2.0), put_bush)
    n_r = scatter(45, lambda x, y: (T.sample(T.stream_d, x, y) < 12 or T.z(x, y) > 0.55 * HEIGHT or rng.random() < 0.2) and clear_of(x, y, 2.5),
                  lambda x, y: (put_ph("rock_moss_set_01" if rng.random() < 0.5 else "rock_moss_set_02", "props", (0.9, 2.2), 0.25))(x, y))
    scatter(3, lambda x, y: clear_of(x, y, 4), put_ph("boulder_01", "props", (1.0, 1.4), 0.3))
    scatter(6, lambda x, y: f_at(x, y) > 0.3 and clear_of(x, y, 2.5), put_ph("tree_stump_01", "props", (0.9, 1.2), 0.1))
    scatter(4, lambda x, y: edge(x, y) and clear_of(x, y, 4) and T.slope(x, y) < 0.15, put_ph("dead_tree_trunk", "props", (0.9, 1.1), 0.15))
    # cover bushes beside targets (partially hidden)
    for t in targets:
        if t["model"] in ("person_plain_stand",):
            continue
        if rng.random() < 0.85:
            a = rng.uniform(0, 2 * math.pi); d = t["r"] + rng.uniform(0.6, 1.4)
            bx, by = t["x"] + d * math.cos(a), t["y"] + d * math.sin(a)
            tile(bx, by, "under").merge(bushes["dark"][int(rng.integers(0, 4))], (bx, by, T.z(bx, by) - 0.1), rng.uniform(0, 6.28), rng.uniform(0.8, 1.1))
    print(f"  undergrowth: {n_f} ferns, {n_s} plants, {n_b} bushes, {n_r} rocks {time.time() - t0:.0f}s", flush=True)

    # ---------------- write vegetation tiles
    for fn in os.listdir(VEG_M) if os.path.exists(VEG_M) else []:
        if fn.startswith("tile_"):
            os.remove(os.path.join(VEG_M, fn))
    tris = 0
    for k in list(tiles):
        i, j, layer = k
        m = tiles[k]
        m.write_glb(os.path.join(VEG_M, f"tile_{i}_{j}_{layer}.glb"), MATS)
        tris += sum(len(f) * (len(f[0]) - 2) for f in m.faces.values()) + sum(sum(len(t) for t in b[3].values()) for b in m.blocks)
        tiles[k] = None   # free memory early (keys are still used for the SDF)
    GA.write_model(os.path.join(MODELS, "forest_veg"), "forest_veg", """<?xml version="1.0"?>
<sdf version="1.9"><model name="forest_veg"><static>true</static><link name="link"/></model></sdf>""")
    print(f"  {len(tiles)} tile meshes, {tris / 1e6:.2f} M triangles {time.time() - t0:.0f}s", flush=True)

    write_terrain(T, trees)
    write_far(T, rng)
    build_targets()

    # ---------------- stream water surface
    wm = Mesh(); wm.material("water")
    xs = np.arange(-HALF - 0.5, HALF + 0.6, 1.0)
    P, UV, tri = [], [], []
    for k, x in enumerate(xs):
        yc = float(stream_y(x))
        zw = min(T.z(x, yc + o) for o in (-1.5, 0, 1.5)) + 0.42
        for o in (-5.5, 5.5):
            P.append((x, yc + o, zw)); UV.append((x / 4, o / 4))
        if k:
            a = 2 * k
            tri += [(a - 2, a - 1, a), (a, a - 1, a + 1)]
    wm.add_arrays("water", P, UV, np.tile([0, 0, 1.0], (len(P), 1)), np.array(tri))
    wm.write_glb(os.path.join(VEG_M, "water.glb"), {"water": dict(color=(0.09, 0.15, 0.13), rough=0.06, double=True)})

    # ---------------- world SDF
    tiles_sdf = ""
    for i in range(NT):
        for j in range(NT):
            vis = ""
            for layer, shadow in (("trees", "true"), ("under", "false"), ("props", "true")):
                if (i, j, layer) in tiles:
                    vis += f"""<visual name="{layer}"><cast_shadows>{shadow}</cast_shadows><geometry><mesh><uri>model://forest_veg/meshes/tile_{i}_{j}_{layer}.glb</uri></mesh></geometry></visual>"""
            if vis:
                tiles_sdf += f"""    <model name="veg_{i}_{j}"><static>true</static><link name="l">{vis}</link></model>\n"""
    tgt_sdf = "".join(f"""    <include><name>target_{t['id']}_{t['model']}</name><uri>model://{t['model']}</uri><pose>{t['x']:.2f} {t['y']:.2f} {t['z']:.2f} 0 0 {t['yaw']:.3f}</pose></include>\n""" for t in targets)
    bz = T.z(*BASE_XY)
    gui = open(os.path.join(WS, "scripts", "gui_block.xml")).read()
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="forest">
    <physics name="4ms" type="ode"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <gravity>0 0 -9.8</gravity>
    <scene>
      <ambient>0.5 0.52 0.55 1</ambient>
      <background>0.62 0.76 0.92 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
      <sky><clouds><speed>4</speed><humidity>0.45</humidity><mean_size>0.6</mean_size></clouds></sky>
    </scene>
{gui}
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 300 0 0 0</pose>
      <diffuse>1.0 0.95 0.86 1</diffuse>
      <specular>0.3 0.3 0.28 1</specular>
      <intensity>1.15</intensity>
      <direction>-0.48 0.42 -0.77</direction>
    </light>

    <include><name>terrain</name><uri>model://forest_terrain</uri></include>
    <include><name>horizon</name><uri>model://forest_far</uri></include>
    <model name="stream"><static>true</static><link name="l"><visual name="v"><cast_shadows>false</cast_shadows><geometry><mesh><uri>model://forest_veg/meshes/water.glb</uri></mesh></geometry></visual></link></model>
    <model name="helipad"><static>true</static><pose>{BASE_XY[0]} {BASE_XY[1]} {bz + 0.02:.2f} 0 0 0</pose><link name="l">
      <visual name="pad"><geometry><cylinder><radius>2.6</radius><length>0.12</length></cylinder></geometry><material><diffuse>0.28 0.28 0.27 1</diffuse><ambient>0.28 0.28 0.27 1</ambient></material></visual>
      <visual name="h1"><pose>0 -0.6 0.065 0 0 0</pose><geometry><box><size>1.8 0.25 0.01</size></box></geometry><material><diffuse>0.95 0.95 0.9 1</diffuse><ambient>0.95 0.95 0.9 1</ambient></material></visual>
      <visual name="h2"><pose>0 0.6 0.065 0 0 0</pose><geometry><box><size>1.8 0.25 0.01</size></box></geometry><material><diffuse>0.95 0.95 0.9 1</diffuse><ambient>0.95 0.95 0.9 1</ambient></material></visual>
      <visual name="h3"><pose>0 0 0.065 0 0 0</pose><geometry><box><size>0.25 1.2 0.01</size></box></geometry><material><diffuse>0.95 0.95 0.9 1</diffuse><ambient>0.95 0.95 0.9 1</ambient></material></visual>
    </link></model>
{tiles_sdf}{tgt_sdf}
    <include><name>camo_drone</name><uri>model://camo_drone</uri><pose>{BASE_XY[0]} {BASE_XY[1]} {bz + 0.6:.2f} 0 0 0.78</pose></include>
  </world>
</sdf>
"""
    with open(os.path.join(WS, "worlds", "forest.sdf"), "w") as f:
        f.write(sdf)
    for t in targets:
        t.pop("r", None)
    with open(os.path.join(WS, "worlds", "targets.json"), "w") as f:
        json.dump(dict(seed=args.seed, world="forest", terrain_size=SIZE, targets=targets), f, indent=2)
    with open(os.path.join(WS, "worlds", "world_info.json"), "w") as f:
        json.dump(dict(world="forest", sdf="forest.sdf", terrain_dir="forest_terrain", size=SIZE, height=HEIGHT,
                       base=list(BASE_XY)), f, indent=2)
    print(f"world written: {len(trees)} trees, {len(targets)} targets, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
