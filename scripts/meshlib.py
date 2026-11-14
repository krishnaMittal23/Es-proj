"""Tiny procedural mesh builder -> Wavefront OBJ/MTL with UVs (for Gazebo)."""
import math, os
import numpy as np


class Mesh:
    def __init__(self):
        self.v, self.vt, self.vn = [], [], []
        self.faces = {}   # material -> list of faces [(vi,ti,ni),...]
        self.materials = {}  # name -> dict(Kd=(r,g,b), map=str|None)

    def material(self, name, kd=(0.5, 0.5, 0.5), tex=None, ks=0.0):
        self.materials[name] = dict(Kd=kd, map=tex, Ks=ks)

    # --- low level -------------------------------------------------
    def _add(self, mat, P, UV, quads_or_tris, flip=False):
        """P: (n,3) array, UV: (n,2), faces idx lists. Smooth normals."""
        P = np.asarray(P, float)
        base_v, base_t, base_n = len(self.v), len(self.vt), len(self.vn)
        N = np.zeros_like(P)
        for f in quads_or_tris:
            a, b, c = P[f[0]], P[f[1]], P[f[2]]
            n = np.cross(b - a, c - a)
            for i in f:
                N[i] += n
        L = np.linalg.norm(N, axis=1, keepdims=True)
        L[L == 0] = 1
        N = N / L
        if flip:
            N = -N
        self.v += [tuple(p) for p in P]
        self.vt += [tuple(u) for u in UV]
        self.vn += [tuple(n) for n in N]
        lst = self.faces.setdefault(mat, [])
        for f in quads_or_tris:
            idx = [(base_v + i + 1, base_t + i + 1, base_n + i + 1) for i in f]
            if flip:
                idx = idx[::-1]
            lst.append(idx)

    # --- primitives ------------------------------------------------
    def ellipsoid(self, mat, c, r, seg=14, rings=8, noise=0.0, rng=None, uvscale=1.0):
        rng = rng or np.random.default_rng(0)
        P, UV = [], []
        for i in range(rings + 1):
            th = math.pi * i / rings
            for j in range(seg + 1):
                ph = 2 * math.pi * j / seg
                k = 1.0 + noise * (rng.random() - 0.5) if 0 < i < rings and j < seg else 1.0
                if j == seg:  # seam must match
                    k = None
                P.append([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph), k])
                UV.append((j / seg * uvscale, i / rings * uvscale))
        P = np.array(P)
        # fix seam noise
        for i in range(rings + 1):
            P[i * (seg + 1) + seg, 3] = P[i * (seg + 1), 3]
        K = P[:, 3:4]
        # Z-up: map (x, y_up, z) -> (x, z, y)
        pts = np.stack([P[:, 0] * K[:, 0] * r[0], P[:, 2] * K[:, 0] * r[1], P[:, 1] * K[:, 0] * r[2]], axis=1) + np.array(c)
        F = []
        for i in range(rings):
            for j in range(seg):
                a = i * (seg + 1) + j
                b = a + seg + 1
                F.append([a, a + 1, b + 1, b])
        self._add(mat, pts, UV, F, flip=True)

    def cylinder(self, mat, p0, p1, r0, r1=None, seg=10, cap=True, uvscale=1.0):
        r1 = r0 if r1 is None else r1
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        d = d / L
        a = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(d, a); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        P, UV = [], []
        for e, (p, r) in enumerate(((p0, r0), (p1, r1))):
            for j in range(seg + 1):
                ph = 2 * math.pi * j / seg
                P.append(p + r * (math.cos(ph) * u + math.sin(ph) * w))
                UV.append((j / seg * uvscale, e * uvscale * max(L, 0.1)))
        F = []
        for j in range(seg):
            F.append([j, j + 1, seg + 1 + j + 1, seg + 1 + j])
        self._add(mat, P, UV, F, flip=True)
        if cap:
            for e, (p, r) in enumerate(((p0, r0), (p1, r1))):
                if r <= 0:
                    continue
                ring = [p + r * (math.cos(2 * math.pi * j / seg) * u + math.sin(2 * math.pi * j / seg) * w) for j in range(seg)]
                P2 = [p] + ring
                UV2 = [(0.5, 0.5)] + [(0.5 + 0.5 * math.cos(2 * math.pi * j / seg), 0.5 + 0.5 * math.sin(2 * math.pi * j / seg)) for j in range(seg)]
                F2 = [[0, 1 + j, 1 + (j + 1) % seg] for j in range(seg)]
                self._add(mat, P2, UV2, F2, flip=(e == 0))

    def box(self, mat, c, s, rot_z=0.0, uvscale=1.0):
        cx, cy, cz = c
        hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
        cs, sn = math.cos(rot_z), math.sin(rot_z)
        def T(x, y, z):
            return [cx + x * cs - y * sn, cy + x * sn + y * cs, cz + z]
        faces = [
            ([(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz)]),
            ([(-hx, -hy, hz), (-hx, hy, hz), (hx, hy, hz), (hx, -hy, hz)]),
            ([(-hx, -hy, -hz), (-hx, -hy, hz), (hx, -hy, hz), (hx, -hy, -hz)]),
            ([(hx, -hy, -hz), (hx, -hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
            ([(hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
            ([(-hx, hy, -hz), (-hx, hy, hz), (-hx, -hy, hz), (-hx, -hy, -hz)]),
        ]
        for q in faces:
            P = [T(*p) for p in q]
            UV = [(0, 0), (uvscale, 0), (uvscale, uvscale), (0, uvscale)]
            self._add(mat, P, UV, [[0, 1, 2, 3]], flip=False)

    def transform(self, scale=1.0, offset=(0, 0, 0)):
        self.v = [tuple(np.array(p) * scale + offset) for p in self.v]

    # --- output ----------------------------------------------------
    def write(self, path):
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        with open(path, "w") as f:
            f.write(f"mtllib {base}.mtl\n")
            for p in self.v:
                f.write("v %.5f %.5f %.5f\n" % p)
            for p in self.vt:
                f.write("vt %.5f %.5f\n" % p)
            for p in self.vn:
                f.write("vn %.5f %.5f %.5f\n" % p)
            for m, fl in self.faces.items():
                f.write(f"usemtl {m}\n")
                for face in fl:
                    f.write("f " + " ".join("%d/%d/%d" % t for t in face) + "\n")
        with open(os.path.join(d, base + ".mtl"), "w") as f:
            for m, a in self.materials.items():
                kd = a["Kd"]
                f.write(f"newmtl {m}\nKa {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\nKd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\n"
                        f"Ks {a['Ks']:.2f} {a['Ks']:.2f} {a['Ks']:.2f}\nNs 10\nd 1\n")
                if a["map"]:
                    f.write(f"map_Kd {a['map']}\n")
                f.write("\n")
