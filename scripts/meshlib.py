"""Tiny procedural mesh builder -> Wavefront OBJ/MTL with UVs (for Gazebo)."""
import math, os
import numpy as np


class Mesh:
    def __init__(self):
        self.v, self.vt, self.vn = [], [], []
        self.faces = {}   # material -> list of faces [(vi,ti,ni),...]
        self.materials = {}  # name -> dict(Kd=(r,g,b), map=str|None)
        self.blocks = []  # compact merged instances: (V, T, N, {mat: (k,3) local tri indices})

    def tri_arrays(self):
        """Cached numpy view of this mesh: V, T, N float32 and per-material 0-based triangle arrays."""
        if getattr(self, "_cache", None) is None:
            tris = {}
            for m, fl in self.faces.items():
                idx = []
                for f in fl:
                    for k in range(1, len(f) - 1):
                        idx.append((f[0][0] - 1, f[k][0] - 1, f[k + 1][0] - 1))
                tris[m] = np.asarray(idx, np.int64).reshape(-1, 3)
            self._cache = (np.asarray(self.v, np.float32).reshape(-1, 3), np.asarray(self.vt, np.float32).reshape(-1, 2),
                           np.asarray(self.vn, np.float32).reshape(-1, 3), tris)
        return self._cache

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

    def card(self, mat, c, w, h, yaw=0.0, pitch=0.0, uv=(0, 0, 1, 1), bend=0.0, normal_up=0.0, roll=0.0):
        """Alpha-cutout billboard quad, bottom-centre at c, facing yaw, tilted by pitch (rad).
        uv=(u0,v0,u1,v1) in image coords (v down). bend curves the card (fronds/grass droop).
        normal_up blends normals toward +Z so foliage is lit like a volume, not flat planes."""
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        right = np.array([-sy, cy, 0.0])
        up = np.array([cy * sp, sy * sp, cp])
        if roll:
            right = right * math.cos(roll) + np.cross(up, right) * math.sin(roll)
        fwd = np.cross(right, up)
        n_rows = 3 if bend else 1
        P, UV = [], []
        for r in range(n_rows + 1):
            t = r / n_rows
            off = up * h * t + fwd * bend * h * t * t
            for s in (-0.5, 0.5):
                P.append(np.asarray(c, float) + right * w * s + off)
                UV.append((uv[0] + (s + 0.5) * (uv[2] - uv[0]), 1.0 - (uv[3] - t * (uv[3] - uv[1]))))
        F = [[2 * r, 2 * r + 1, 2 * r + 3, 2 * r + 2] for r in range(n_rows)]
        base = len(self.v)
        self._add(mat, P, UV, F)
        if normal_up:
            for i in range(base, len(self.v)):
                n = np.array(self.vn[i]) * (1 - normal_up) + np.array([0, 0, 1.0]) * normal_up
                n = n + (np.array(self.v[i]) - np.asarray(c)) * 0.15 * normal_up
                self.vn[i] = tuple(n / (np.linalg.norm(n) + 1e-9))

    def add_arrays(self, mat, P, UV, N, tris):
        """Fast path: numpy positions/uvs (OBJ convention, v up)/normals + (k,3) triangle indices."""
        b = len(self.v)
        self.v += [tuple(p) for p in np.asarray(P, float)]
        self.vt += [tuple(u) for u in np.asarray(UV, float)]
        self.vn += [tuple(n) for n in np.asarray(N, float)]
        self.faces.setdefault(mat, []).extend([[(b + i + 1,) * 3 for i in t] for t in np.asarray(tris) .tolist()])

    def merge(self, other, offset=(0, 0, 0), yaw=0.0, scale=1.0):
        """Append another Mesh (materials merged by name) with transform."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float32)
        V0, T0, N0, tris = other.tri_arrays()
        V = (V0 * np.float32(scale)) @ R.T + np.asarray(offset, np.float32)
        for m, a in other.materials.items():
            self.materials.setdefault(m, a)
        self.blocks.append((V.astype(np.float32), T0, (N0 @ R.T).astype(np.float32), tris))

    def write_glb(self, path, mats=None):
        """Write binary glTF. mats: name -> dict(tex=path, normal=path, rough=0.9, alpha=False,
        double=False, color=(r,g,b)). Textures are embedded (self-contained model)."""
        import json, struct
        mats = mats or {}
        base = len(self.v)
        Vs, Ts, Ns = [np.asarray(self.v, np.float32).reshape(-1, 3)], [np.asarray(self.vt, np.float32).reshape(-1, 2)], [np.asarray(self.vn, np.float32).reshape(-1, 3)]
        block_tris = {}
        for bV, bT, bN, btris in self.blocks:
            for m, t in btris.items():
                if len(t):
                    block_tris.setdefault(m, []).append(t + base)
            Vs.append(bV); Ts.append(bT); Ns.append(bN); base += len(bV)
        V, T, N = np.concatenate(Vs), np.concatenate(Ts).copy(), np.concatenate(Ns)
        del Vs, Ts, Ns
        T[:, 1] = 1.0 - T[:, 1]          # OBJ (v up) -> glTF (v down)
        blob = bytearray(); views = []; accs = []

        def add_view(data, target=None):
            while len(blob) % 4:
                blob.append(0)
            off = len(blob); blob.extend(data)
            v = dict(buffer=0, byteOffset=off, byteLength=len(data))
            if target:
                v["target"] = target
            views.append(v); return len(views) - 1

        def add_acc(arr, typ, comp, target, minmax=False):
            vi = add_view(arr.tobytes(), target)
            a = dict(bufferView=vi, componentType=comp, count=int(arr.shape[0]), type=typ)
            if minmax:
                a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
            accs.append(a); return len(accs) - 1

        ap = add_acc(V, "VEC3", 5126, 34962, True)
        an = add_acc(N, "VEC3", 5126, 34962)
        at = add_acc(T, "VEC2", 5126, 34962)
        images, textures, materials, prims = [], [], [], []
        img_idx = {}

        def tex(p):
            if p not in img_idx:
                if p.startswith("uri:"):          # external file, relative to the .glb (shared GPU texture)
                    images.append(dict(uri=p[4:]))
                else:
                    vi = add_view(open(p, "rb").read())
                    images.append(dict(bufferView=vi, mimeType="image/png" if p.endswith(".png") else "image/jpeg"))
                textures.append(dict(source=len(images) - 1, sampler=0))
                img_idx[p] = len(textures) - 1
            return img_idx[p]

        for name in list(dict.fromkeys(list(self.faces) + list(block_tris))):
            idx = []
            for f in self.faces.get(name, []):
                for k in range(1, len(f) - 1):
                    idx += [f[0][0] - 1, f[k][0] - 1, f[k + 1][0] - 1]
            parts = [np.asarray(idx, np.int64)] + [t.ravel() for t in block_tris.get(name, [])]
            idx = np.concatenate(parts)
            if not len(idx):
                continue
            ai = add_acc(idx.astype(np.uint32), "SCALAR", 5125, 34963)
            o = mats.get(name, {})
            kd = o.get("color", self.materials.get(name, {}).get("Kd", (0.8, 0.8, 0.8)))
            pbr = dict(baseColorFactor=[float(kd[0]), float(kd[1]), float(kd[2]), 1.0],
                       metallicFactor=0.0, roughnessFactor=float(o.get("rough", 0.9)))
            if o.get("tex"):
                pbr["baseColorTexture"] = dict(index=tex(o["tex"]))
                pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0] if "color" not in o else pbr["baseColorFactor"]
            m = dict(name=name, pbrMetallicRoughness=pbr, doubleSided=bool(o.get("double", False)))
            if o.get("normal"):
                m["normalTexture"] = dict(index=tex(o["normal"]))
            if o.get("alpha"):
                m["alphaMode"] = "MASK"; m["alphaCutoff"] = float(o.get("cutoff", 0.5))
            materials.append(m)
            prims.append(dict(attributes=dict(POSITION=ap, NORMAL=an, TEXCOORD_0=at), indices=ai, material=len(materials) - 1))

        gl = dict(asset=dict(version="2.0", generator="camo meshlib"), scene=0, scenes=[dict(nodes=[0])],
                  nodes=[dict(mesh=0)], meshes=[dict(primitives=prims)], accessors=accs, bufferViews=views,
                  buffers=[dict(byteLength=0)], materials=materials)
        if images:
            gl.update(images=images, textures=textures, samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)])
        while len(blob) % 4:
            blob.append(0)
        gl["buffers"][0]["byteLength"] = len(blob)
        js = json.dumps(gl, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(bytes(blob))

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
