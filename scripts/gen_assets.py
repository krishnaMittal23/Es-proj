#!/usr/bin/env python3
"""Generate all procedural assets for the Savanna camouflage world.

Outputs into  <ws>/models/<name>/  (model.config, model.sdf, meshes/, textures/)
and returns terrain sampler used by gen_world.py.
"""
import os, math, json
import numpy as np
from PIL import Image
from meshlib import Mesh

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(WS, "models")

TERRAIN_SIZE = 240.0     # metres (x and y)
TERRAIN_H = 14.0         # metres of vertical relief
HM_N = 513               # 2^n + 1


# ---------------------------------------------------------------- noise
def fbm(n, octaves=6, seed=0, persistence=0.5):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        res = 4 * 2 ** o
        g = rng.random((res + 1, res + 1))
        xs = np.linspace(0, res, n)
        x0 = np.floor(xs).astype(int).clip(0, res - 1)
        t = xs - x0
        t = t * t * (3 - 2 * t)
        a = g[np.ix_(x0, x0)] * (1 - t)[None, :] * (1 - t)[:, None]
        b = g[np.ix_(x0, x0 + 1)] * t[None, :] * (1 - t)[:, None]
        c = g[np.ix_(x0 + 1, x0)] * (1 - t)[None, :] * t[:, None]
        d = g[np.ix_(x0 + 1, x0 + 1)] * t[None, :] * t[:, None]
        out += amp * (a + b + c + d)
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def tileable_noise(n, seed, octaves=5):
    """Wrap-around fbm for seamless textures."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n)); amp = 1; tot = 0
    for o in range(octaves):
        res = 4 * 2 ** o
        g = rng.random((res, res))
        xs = np.linspace(0, res, n, endpoint=False)
        x0 = np.floor(xs).astype(int) % res
        x1 = (x0 + 1) % res
        t = xs - np.floor(xs); t = t * t * (3 - 2 * t)
        a = g[np.ix_(x0, x0)] * (1 - t)[None, :] * (1 - t)[:, None]
        b = g[np.ix_(x0, x1)] * t[None, :] * (1 - t)[:, None]
        c = g[np.ix_(x1, x0)] * (1 - t)[None, :] * t[:, None]
        d = g[np.ix_(x1, x1)] * t[None, :] * t[:, None]
        out += amp * (a + b + c + d); tot += amp; amp *= 0.55
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def save_img(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def colorize(noise, stops):
    """stops: list of (t, (r,g,b))"""
    ts = [s[0] for s in stops]
    out = np.zeros(noise.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(noise, ts, [s[1][c] for s in stops])
    return out


# ---------------------------------------------------------------- terrain
class Terrain:
    def __init__(self):
        n = HM_N
        base = fbm(n, 6, seed=7, persistence=0.48)
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
        # one gentle hill (north-east) and a shallow dry riverbed meandering across
        hill = np.exp(-(((xx - 0.78) ** 2 + (yy - 0.25) ** 2) / 0.018))
        river_c = 0.55 + 0.12 * np.sin(xx * 7.0) + 0.04 * np.sin(xx * 19.0)
        river = np.exp(-((yy - river_c) ** 2) / 0.0014)
        h = 0.35 * base + 0.55 * hill - 0.22 * river
        # flatten centre search area a bit so targets sit nicely
        h = h * 0.9
        h = (h - h.min()) / (h.max() - h.min())
        self.h = h
        self.river = river

    def png(self, path):
        save_img(self.h * 255.0, path)
        # 16-bit would be nicer; 8-bit w/ 14 m relief = 5.5 cm steps, acceptable

    def z(self, x, y):
        """World height at (x,y). Heightmap model is placed with z offset 0."""
        n = HM_N
        u = (x / TERRAIN_SIZE + 0.5) * (n - 1)
        v = (0.5 - y / TERRAIN_SIZE) * (n - 1)  # image row 0 = +y
        u = min(max(u, 0), n - 1.001); v = min(max(v, 0), n - 1.001)
        i, j = int(v), int(u)
        fu, fv = u - j, v - i
        h = (self.h[i, j] * (1 - fu) * (1 - fv) + self.h[i, j + 1] * fu * (1 - fv)
             + self.h[i + 1, j] * (1 - fu) * fv + self.h[i + 1, j + 1] * fu * fv)
        return float(h * TERRAIN_H)

    def riverness(self, x, y):
        n = HM_N
        u = int(min(max((x / TERRAIN_SIZE + 0.5) * (n - 1), 0), n - 1))
        v = int(min(max((0.5 - y / TERRAIN_SIZE) * (n - 1), 0), n - 1))
        return float(self.river[v, u])


def make_terrain_model(T):
    d = os.path.join(MODELS, "savanna_terrain")
    T.png(os.path.join(d, "textures", "heightmap.png"))
    S = 512
    # dry grass: golden / olive / straw mix with fine speckle
    n1, n2 = tileable_noise(S, 1), tileable_noise(S, 2, 6)
    speck = np.random.default_rng(3).random((S, S))
    grass = colorize(0.6 * n1 + 0.4 * n2, [(0, (118, 100, 52)), (0.4, (150, 128, 68)), (0.7, (176, 150, 84)), (1, (196, 172, 102))])
    grass *= (0.88 + 0.24 * speck)[..., None]
    save_img(grass, os.path.join(d, "textures", "grass_diffuse.png"))
    dirt = colorize(0.7 * tileable_noise(S, 4) + 0.3 * n2, [(0, (112, 78, 52)), (0.5, (146, 104, 70)), (1, (178, 134, 94))])
    dirt *= (0.9 + 0.2 * speck)[..., None]
    save_img(dirt, os.path.join(d, "textures", "dirt_diffuse.png"))
    sand = colorize(tileable_noise(S, 5), [(0, (200, 176, 132)), (1, (226, 206, 160))]) * (0.94 + 0.12 * speck)[..., None]
    save_img(sand, os.path.join(d, "textures", "sand_diffuse.png"))
    # normal maps from noise
    for name, seed in (("grass", 11), ("dirt", 12), ("sand", 13)):
        hgt = tileable_noise(S, seed, 6)
        gx = np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)
        gy = np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)
        nx, ny, nz = -gx * 1.5, -gy * 1.5, np.ones_like(hgt)
        L = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        nm = np.stack([nx / L, ny / L, nz / L], -1) * 0.5 + 0.5
        save_img(nm * 255, os.path.join(d, "textures", f"{name}_normal.png"))
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="savanna_terrain">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><heightmap>
          <uri>model://savanna_terrain/textures/heightmap.png</uri>
          <size>{TERRAIN_SIZE} {TERRAIN_SIZE} {TERRAIN_H}</size>
          <pos>0 0 0</pos>
        </heightmap></geometry>
      </collision>
      <visual name="visual">
        <geometry><heightmap>
          <use_terrain_paging>false</use_terrain_paging>
          <texture>
            <diffuse>model://savanna_terrain/textures/sand_diffuse.png</diffuse>
            <normal>model://savanna_terrain/textures/sand_normal.png</normal>
            <size>6</size>
          </texture>
          <texture>
            <diffuse>model://savanna_terrain/textures/dirt_diffuse.png</diffuse>
            <normal>model://savanna_terrain/textures/dirt_normal.png</normal>
            <size>7</size>
          </texture>
          <texture>
            <diffuse>model://savanna_terrain/textures/grass_diffuse.png</diffuse>
            <normal>model://savanna_terrain/textures/grass_normal.png</normal>
            <size>5</size>
          </texture>
          <blend><min_height>{0.03 * TERRAIN_H:.2f}</min_height><fade_dist>{0.04 * TERRAIN_H:.2f}</fade_dist></blend>
          <blend><min_height>{0.10 * TERRAIN_H:.2f}</min_height><fade_dist>{0.10 * TERRAIN_H:.2f}</fade_dist></blend>
          <uri>model://savanna_terrain/textures/heightmap.png</uri>
          <size>{TERRAIN_SIZE} {TERRAIN_SIZE} {TERRAIN_H}</size>
          <pos>0 0 0</pos>
        </heightmap></geometry>
      </visual>
    </link>
  </model>
</sdf>"""
    write_model(d, "savanna_terrain", sdf)


def write_model(d, name, sdf):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.sdf"), "w") as f:
        f.write(sdf)
    with open(os.path.join(d, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model><name>{name}</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>
<description>Procedurally generated asset for camouflage detection project</description></model>""")


def static_mesh_model(name, mesh_file, collision=None, label=None, scale=1.0):
    """SDF for a static mesh model. collision: ('box',sx,sy,sz,cz) or None."""
    col = ""
    if collision:
        _, sx, sy, sz, cz = collision
        col = f"""<collision name="c"><pose>0 0 {cz} 0 0 0</pose><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>"""
    lab = f"""<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin>""" if label is not None else ""
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      {col}
      <visual name="v"><geometry><mesh><uri>model://{name}/meshes/{mesh_file}</uri><scale>{scale} {scale} {scale}</scale></mesh></geometry></visual>
    </link>
    {lab}
  </model>
</sdf>"""


# ---------------------------------------------------------------- textures
def make_foliage_textures(d):
    S = 256
    sp = np.random.default_rng(21).random((S, S))
    bark = colorize(tileable_noise(S, 31, 6), [(0, (62, 48, 36)), (1, (112, 92, 72))]) * (0.85 + 0.3 * sp)[..., None]
    save_img(bark, os.path.join(d, "bark.png"))
    leaf = colorize(tileable_noise(S, 32, 6), [(0, (60, 82, 36)), (0.5, (86, 112, 48)), (1, (124, 142, 64))]) * (0.8 + 0.4 * sp)[..., None]
    save_img(leaf, os.path.join(d, "leaf.png"))
    dry = colorize(tileable_noise(S, 33, 6), [(0, (128, 112, 62)), (1, (186, 164, 96))]) * (0.85 + 0.3 * sp)[..., None]
    save_img(dry, os.path.join(d, "dry.png"))
    rock = colorize(tileable_noise(S, 34, 6), [(0, (96, 88, 80)), (1, (170, 156, 138))]) * (0.85 + 0.3 * sp)[..., None]
    save_img(rock, os.path.join(d, "rock.png"))


def make_camo_textures(d):
    """Camouflage patterns: savanna camo (tan/olive/brown), woodland, desert."""
    S = 256
    pats = {
        "camo_savanna": [(0.0, (112, 98, 56)), (0.38, (150, 128, 72)), (0.55, (84, 92, 48)), (0.78, (170, 146, 92)), (1.0, (96, 76, 48))],
        "camo_woodland": [(0.0, (52, 66, 36)), (0.35, (84, 96, 52)), (0.55, (110, 90, 56)), (0.8, (58, 48, 34)), (1.0, (92, 110, 62))],
        "camo_desert": [(0.0, (176, 150, 104)), (0.4, (196, 172, 124)), (0.62, (150, 124, 84)), (0.85, (208, 186, 140)), (1.0, (130, 108, 72))],
    }
    for k, st in pats.items():
        n = tileable_noise(S, hash(k) % 1000, 5)
        # posterize into blotches => classic camouflage look
        q = np.floor(n * 5) / 4.0
        img = colorize(q, st)
        sp = np.random.default_rng(5).random((S, S))
        img *= (0.93 + 0.14 * sp)[..., None]
        save_img(img, os.path.join(d, k + ".png"))
    # NON-camouflage (bright) clothing for the contrast experiment
    plain = np.zeros((S, S, 3)); plain[..., 0] = 220; plain[..., 1] = 60; plain[..., 2] = 40
    save_img(plain, os.path.join(d, "plain_red.png"))
    plain2 = np.zeros((S, S, 3)); plain2[..., 0] = 40; plain2[..., 1] = 90; plain2[..., 2] = 200
    save_img(plain2, os.path.join(d, "plain_blue.png"))


# ---------------------------------------------------------------- vegetation meshes
def acacia(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    m = Mesh()
    m.material("bark", (0.4, 0.32, 0.25), "bark.png")
    m.material("leaf", (0.4, 0.5, 0.25), "leaf.png")
    h = 5.0 + 2.0 * rng.random()
    lean = rng.uniform(-0.6, 0.6, 2)
    top = np.array([lean[0], lean[1], h])
    m.cylinder("bark", (0, 0, -0.3), (lean[0] * .3, lean[1] * .3, h * 0.55), 0.34, 0.2, seg=9)
    m.cylinder("bark", (lean[0] * .3, lean[1] * .3, h * 0.55), top, 0.2, 0.12, seg=9)
    nb = 4 + int(rng.integers(0, 3))
    for i in range(nb):
        a = 2 * math.pi * i / nb + rng.random()
        L = 2.4 + 1.6 * rng.random()
        end = top + np.array([math.cos(a) * L, math.sin(a) * L, 0.5 + 0.8 * rng.random()])
        m.cylinder("bark", top - [0, 0, 0.3], end, 0.1, 0.05, seg=6)
        # umbrella clump
        for k in range(3):
            c = end + np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, .5)])
            m.ellipsoid("leaf", c, (1.9 + rng.random(), 1.9 + rng.random(), 0.55), seg=10, rings=6, noise=0.35, rng=rng, uvscale=2)
    m.ellipsoid("leaf", top + [0, 0, 0.6], (2.6, 2.6, 0.7), seg=12, rings=6, noise=0.3, rng=rng, uvscale=2)
    m.transform(scale)
    return m


def baobab(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    m = Mesh()
    m.material("bark", (0.45, 0.4, 0.34), "bark.png")
    m.material("leaf", (0.4, 0.5, 0.25), "leaf.png")
    h = 8.0 + 2 * rng.random()
    # bottle trunk from rings
    m.cylinder("bark", (0, 0, -0.4), (0, 0, 1.0), 1.9, 1.7, seg=14, uvscale=2)
    m.cylinder("bark", (0, 0, 1.0), (0, 0, h * 0.6), 1.7, 1.45, seg=14, uvscale=2)
    m.cylinder("bark", (0, 0, h * 0.6), (0, 0, h), 1.45, 0.8, seg=14, uvscale=2)
    nb = 7
    for i in range(nb):
        a = 2 * math.pi * i / nb + rng.random() * .4
        L = 3 + 2 * rng.random()
        mid = np.array([0, 0, h - 0.3]) + [math.cos(a) * L * .5, math.sin(a) * L * .5, L * .4]
        end = np.array([0, 0, h - 0.3]) + [math.cos(a) * L, math.sin(a) * L, L * .55 + 0.5]
        m.cylinder("bark", (0, 0, h - 0.3), mid, 0.4, 0.28, seg=7)
        m.cylinder("bark", mid, end, 0.28, 0.12, seg=7)
        m.ellipsoid("leaf", end + [0, 0, 0.4], (1.4, 1.4, 0.7), seg=9, rings=5, noise=0.4, rng=rng, uvscale=2)
    m.transform(scale)
    return m


def bush(seed, dry=False, scale=1.0):
    rng = np.random.default_rng(seed)
    m = Mesh()
    m.material("leaf", (0.4, 0.5, 0.25), "dry.png" if dry else "leaf.png")
    n = 4 + int(rng.integers(0, 4))
    for _ in range(n):
        c = (rng.uniform(-.9, .9), rng.uniform(-.9, .9), rng.uniform(0.3, 0.8))
        r = rng.uniform(.5, .95)
        m.ellipsoid("leaf", c, (r, r, r * 0.8), seg=9, rings=6, noise=0.45, rng=rng, uvscale=1.5)
    m.transform(scale)
    return m


def rock(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    m = Mesh()
    m.material("rock", (0.5, 0.45, 0.4), "rock.png")
    r = rng.uniform(0.8, 1.4, 3)
    m.ellipsoid("rock", (0, 0, r[2] * 0.35), tuple(r), seg=10, rings=7, noise=0.5, rng=rng, uvscale=2)
    m.transform(scale)
    return m


def termite_mound(seed):
    m = Mesh()
    m.material("rock", (0.5, 0.35, 0.25), "rock.png")
    m.cylinder("rock", (0, 0, -0.1), (0, 0, 1.0), 0.55, 0.35, seg=10)
    m.cylinder("rock", (0, 0, 1.0), (0, 0, 1.8), 0.35, 0.0, seg=10, cap=False)
    return m


def log(seed):
    m = Mesh()
    m.material("bark", (0.4, 0.32, 0.25), "bark.png")
    m.cylinder("bark", (-1.6, 0, 0.3), (1.6, 0, 0.3), 0.3, 0.26, seg=8)
    return m


def grass_clump(seed):
    rng = np.random.default_rng(seed)
    m = Mesh()
    m.material("dry", (0.7, 0.6, 0.35), "dry.png")
    for _ in range(6):
        c = (rng.uniform(-.4, .4), rng.uniform(-.4, .4), 0.18)
        m.ellipsoid("dry", c, (0.35, 0.35, 0.4), seg=7, rings=4, noise=0.5, rng=rng)
    return m


# ---------------------------------------------------------------- characters / targets
def humanoid(pose, cloth_tex, skin=(0.55, 0.38, 0.28), cloth_kd=(0.4, 0.4, 0.3), hat=True):
    """Low-poly humanoid, Z-up, feet at z=0 (standing) or lying on ground (prone), facing +X."""
    m = Mesh()
    m.material("cloth", cloth_kd, cloth_tex)
    m.material("skin", skin)
    m.material("dark", (0.1, 0.1, 0.1))
    rng = np.random.default_rng(1)

    def E(mat, c, r, **k):
        m.ellipsoid(mat, c, r, seg=10, rings=7, **k)

    def part(p, stand_pts):
        return stand_pts

    # define in standing coords then rotate for pose
    pts = []  # list of (kind, args)
    # legs
    pts.append(("cyl", "cloth", (0, -0.11, 0.05), (0, -0.11, 0.86), 0.075, 0.09))
    pts.append(("cyl", "cloth", (0, 0.11, 0.05), (0, 0.11, 0.86), 0.075, 0.09))
    pts.append(("ell", "dark", (0.05, -0.11, 0.05), (0.13, 0.07, 0.05)))
    pts.append(("ell", "dark", (0.05, 0.11, 0.05), (0.13, 0.07, 0.05)))
    # torso
    pts.append(("ell", "cloth", (0, 0, 1.12), (0.14, 0.22, 0.32)))
    pts.append(("ell", "cloth", (0, 0, 0.92), (0.15, 0.2, 0.15)))
    # arms
    if pose == "stand":
        pts.append(("cyl", "cloth", (0, -0.27, 1.38), (0.03, -0.31, 0.84), 0.055, 0.05))
        pts.append(("cyl", "cloth", (0, 0.27, 1.38), (0.03, 0.31, 0.84), 0.055, 0.05))
        pts.append(("ell", "skin", (0.03, -0.31, 0.8), (0.05, 0.05, 0.07)))
        pts.append(("ell", "skin", (0.03, 0.31, 0.8), (0.05, 0.05, 0.07)))
    else:  # arms forward
        pts.append(("cyl", "cloth", (0, -0.27, 1.38), (0.38, -0.16, 1.28), 0.055, 0.05))
        pts.append(("cyl", "cloth", (0, 0.27, 1.38), (0.38, 0.1, 1.28), 0.055, 0.05))
        pts.append(("ell", "skin", (0.4, -0.16, 1.28), (0.05, 0.05, 0.06)))
        pts.append(("ell", "skin", (0.4, 0.1, 1.28), (0.05, 0.05, 0.06)))
    # neck + head
    pts.append(("cyl", "skin", (0, 0, 1.4), (0, 0, 1.52), 0.05, 0.05))
    pts.append(("ell", "skin", (0, 0, 1.65), (0.1, 0.09, 0.12)))
    if hat:
        pts.append(("ell", "cloth", (0, 0, 1.74), (0.12, 0.12, 0.06)))

    def tf(p):
        p = np.array(p, float)
        if pose in ("stand", "aim"):
            return p
        if pose == "crouch":  # fold at hips/knees: squash height and lean forward
            q = p.copy(); q[2] = p[2] * 0.62; q[0] = p[0] + 0.35 * p[2] * 0.6; return q
        if pose == "prone":  # rotate about Y by 90deg (head toward +X)
            return np.array([p[2], p[1], 0.16 - p[0] * 0.5])
        return p

    for kind, mat, *a in pts:
        if kind == "cyl":
            m.cylinder(mat, tf(a[0]), tf(a[1]), a[2], a[3], seg=8)
        else:
            c = tf(a[0]); r = a[1]
            if pose == "prone":
                r = (r[2], r[1], r[0] * 1.2)
            elif pose == "crouch":
                r = (r[0], r[1], r[2] * 0.7)
            m.ellipsoid(mat, c, r, seg=9, rings=6)
    return m


def animal(kind):
    m = Mesh()
    if kind == "rhino":
        m.material("hide", (0.45, 0.42, 0.38), "rock.png")
        m.material("horn", (0.75, 0.7, 0.62))
        m.ellipsoid("hide", (0, 0, 1.15), (1.7, 0.8, 0.85), seg=14, rings=9)
        m.ellipsoid("hide", (1.7, 0, 0.95), (0.7, 0.45, 0.45), seg=12, rings=8)
        m.ellipsoid("hide", (0.9, 0, 1.45), (0.5, 0.45, 0.35), seg=10, rings=7)  # hump
        m.cylinder("horn", (2.15, 0, 1.15), (2.45, 0, 1.75), 0.1, 0.0, seg=8, cap=False)
        m.cylinder("horn", (2.0, 0, 1.15), (2.15, 0, 1.45), 0.07, 0.0, seg=8, cap=False)
        for x in (-1.0, 1.0):
            for y in (-0.45, 0.45):
                m.cylinder("hide", (x, y, 0.0), (x, y, 0.75), 0.17, 0.22, seg=8)
        m.ellipsoid("hide", (1.5, 0.45, 1.3), (0.15, 0.08, 0.2), seg=6, rings=4)
        m.ellipsoid("hide", (1.5, -0.45, 1.3), (0.15, 0.08, 0.2), seg=6, rings=4)
    elif kind == "hippo":
        m.material("hide", (0.45, 0.4, 0.4), "rock.png")
        m.ellipsoid("hide", (0, 0, 0.9), (1.6, 0.9, 0.8), seg=14, rings=9)
        m.ellipsoid("hide", (1.7, 0, 0.9), (0.7, 0.55, 0.5), seg=12, rings=8)
        for x in (-0.9, 0.9):
            for y in (-0.5, 0.5):
                m.cylinder("hide", (x, y, 0.0), (x, y, 0.5), 0.2, 0.26, seg=8)
    return m


def tent(tex):
    m = Mesh()
    m.material("cloth", (0.4, 0.4, 0.3), tex)
    # A-frame tent
    P = [[-1.6, -1.2, 0], [1.6, -1.2, 0], [1.6, 1.2, 0], [-1.6, 1.2, 0], [-1.6, 0, 1.4], [1.6, 0, 1.4]]
    UV = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 1), (2, 1)]
    F = [[0, 1, 5, 4], [3, 4, 5, 2], [0, 4, 3], [1, 2, 5]]
    m._add("cloth", P, UV, F, flip=True)
    return m


def jeep(tex):
    m = Mesh()
    m.material("body", (0.4, 0.4, 0.3), tex)
    m.material("dark", (0.07, 0.07, 0.07))
    m.material("glass", (0.15, 0.2, 0.22))
    m.box("body", (0, 0, 0.85), (4.0, 1.8, 0.8), uvscale=2)
    m.box("body", (-0.2, 0, 1.45), (1.8, 1.7, 0.7), uvscale=1)
    m.box("glass", (0.75, 0, 1.45), (0.1, 1.5, 0.55))
    m.box("body", (1.6, 0, 1.15), (1.0, 1.7, 0.1), uvscale=1)
    for x in (-1.3, 1.3):
        for y in (-0.95, 0.95):
            m.cylinder("dark", (x, y - 0.12, 0.45), (x, y + 0.12, 0.45), 0.45, 0.45, seg=12)
    return m


def drone_mesh():
    return None


# ---------------------------------------------------------------- orchestration
def build_all():
    T = Terrain()
    make_terrain_model(T)

    tex_dir_all = {}

    def emit(name, mesh, textures_src, collision=None, label=None):
        d = os.path.join(MODELS, name)
        mesh.write(os.path.join(d, "meshes", name + ".obj"))
        for t in textures_src:
            for fn in os.listdir(t):
                if fn.endswith(".png"):
                    Image.open(os.path.join(t, fn)).save(os.path.join(d, "meshes", fn))
        write_model(d, name, static_mesh_model(name, name + ".obj", collision, label))

    tmp_f = os.path.join(WS, "build", "tex_foliage"); make_foliage_textures(tmp_f)
    tmp_c = os.path.join(WS, "build", "tex_camo"); make_camo_textures(tmp_c)

    for i in range(4):
        emit(f"acacia_{i}", acacia(100 + i), [tmp_f], ("box", 1.0, 1.0, 6.0, 3.0))
    for i in range(2):
        emit(f"baobab_{i}", baobab(200 + i), [tmp_f], ("box", 3.4, 3.4, 8.0, 4.0))
    for i in range(3):
        emit(f"bush_{i}", bush(300 + i), [tmp_f])
    for i in range(2):
        emit(f"drybush_{i}", bush(400 + i, dry=True), [tmp_f])
    for i in range(3):
        emit(f"rock_{i}", rock(500 + i), [tmp_f], ("box", 2.0, 2.0, 1.0, 0.5))
    emit("termite_mound", termite_mound(0), [tmp_f])
    emit("fallen_log", log(0), [tmp_f])
    for i in range(2):
        emit(f"grass_clump_{i}", grass_clump(600 + i), [tmp_f])

    # LABELS (used by bounding-box camera): 1=person 2=vehicle 3=animal ; tent counted as vehicle/object class 2
    # camouflaged people
    emit("person_camo_stand", humanoid("stand", "camo_savanna.png"), [tmp_c], label=1)
    emit("person_camo_crouch", humanoid("crouch", "camo_woodland.png"), [tmp_c], label=1)
    emit("person_camo_prone", humanoid("prone", "camo_desert.png"), [tmp_c], label=1)
    # NON camo (control group)
    emit("person_plain_stand", humanoid("stand", "plain_red.png"), [tmp_c], label=1)
    emit("person_plain_prone", humanoid("prone", "plain_blue.png"), [tmp_c], label=1)
    emit("tent_camo", tent("camo_woodland.png"), [tmp_c], label=2)
    emit("jeep_camo", jeep("camo_savanna.png"), [tmp_c], label=2)
    emit("rhino", animal("rhino"), [tmp_f], label=3)
    emit("hippo", animal("hippo"), [tmp_f], label=3)
    return T


if __name__ == "__main__":
    build_all()
    print("assets written to", MODELS)
