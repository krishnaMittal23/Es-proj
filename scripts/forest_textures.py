#!/usr/bin/env python3
"""Texture generation for the forest world.

Foliage atlases are painted procedurally with colour sampled from photo-scanned leaves; ground,
bark and grass use CC0 Poly Haven photo textures (assets_src/, see fetch_assets.py).
All RGBA atlases get colour-bleeding into transparent texels so mip-mapping gives no dark halos.
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(WS, "assets_src")


def bleed(img):
    """Fill RGB of transparent pixels with nearby opaque colour (prevents dark mip fringes)."""
    a = np.asarray(img, np.float32) / 255.0
    rgb, al = a[..., :3], a[..., 3]
    out = rgb.copy()
    filled = al > 0.5
    pre = np.concatenate([rgb * al[..., None], al[..., None]], -1)
    for r in (2, 6, 16, 40, 120):
        b = np.stack([np.asarray(Image.fromarray((pre[..., i] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255.0
                      for i in range(4)], -1)
        c = b[..., :3] / np.maximum(b[..., 3:4], 1e-4)
        new = (~filled) & (b[..., 3] > 0.003)
        out[new] = np.clip(c[new], 0, 1)
        filled |= new
    if (~filled).any():
        out[~filled] = rgb[al > 0.5].mean(0) if (al > 0.5).any() else 0.3
    return Image.fromarray((np.concatenate([out, al[..., None]], -1) * 255).clip(0, 255).astype(np.uint8), "RGBA")


def _jit(c, rng, s=14):
    return tuple(int(np.clip(v + rng.normal(0, s), 0, 255)) for v in c)


def leaf_shape(cx, cy, ang, L, W, tip=0.8):
    pts_l, pts_r = [], []
    ca, sa = math.cos(ang), math.sin(ang)
    for k in range(13):
        t = k / 12
        w = W * (math.sin(math.pi * t) ** tip) * (1 - 0.25 * t)
        ax, ay = cx + ca * L * t, cy + sa * L * t
        pts_l.append((ax - sa * w, ay + ca * w)); pts_r.append((ax + sa * w, ay - ca * w))
    return pts_l, pts_r


def broadleaf_atlas(path, palette, seed, S=1024, leaf_len=(46, 78), density=1.0):
    """2x2 cells of leafy sprays, stem from bottom-centre to top."""
    rng = np.random.default_rng(seed)
    K = 2 * S
    im = Image.new("RGBA", (K, K), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    C = K // 2
    for cell in range(4):
        ox, oy = (cell % 2) * C, (cell // 2) * C
        # main stem + side twigs
        stems = []
        x0, y0 = ox + C * 0.5, oy + C * 0.98
        x1, y1 = ox + C * (0.5 + rng.uniform(-0.12, 0.12)), oy + C * 0.06
        stems.append(((x0, y0), (x1, y1)))
        for k in range(int(rng.integers(4, 7))):
            t = rng.uniform(0.15, 0.8)
            sx, sy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            side = rng.choice([-1, 1]); L = C * rng.uniform(0.25, 0.42) * (1.1 - t * 0.5)
            a = -math.pi / 2 + side * rng.uniform(0.5, 1.0)
            stems.append(((sx, sy), (sx + math.cos(a) * L, sy + math.sin(a) * L)))
        for (ax, ay), (bx, by) in stems:
            dr.line([(ax, ay), (bx, by)], fill=(88, 72, 48, 255), width=5)
        # leaves along all stems
        n_leaves = int(95 * density)
        for k in range(n_leaves):
            (ax, ay), (bx, by) = stems[int(rng.integers(0, len(stems)))]
            t = rng.uniform(0.05, 1.0)
            px, py = ax + (bx - ax) * t, ay + (by - ay) * t
            sa = math.atan2(by - ay, bx - ax)
            ang = sa + rng.choice([-1, 1]) * rng.uniform(0.4, 1.3)
            L = rng.uniform(*leaf_len) * 2; W = L * rng.uniform(0.24, 0.32)
            base = palette[int(rng.integers(0, len(palette)))]
            col = _jit(base, rng, 10)
            pl, pr = leaf_shape(px, py, ang, L, W)
            dark = tuple(int(c * 0.78) for c in col)
            dr.polygon(pl + pr[::-1], fill=col + (255,))
            dr.polygon(pl + [pl[-1]] + [(px + math.cos(ang) * L, py + math.sin(ang) * L)] + [(px, py)], fill=dark + (255,))
            light = tuple(min(255, int(c * 1.25)) for c in col)
            dr.line([(px, py), (px + math.cos(ang) * L * 0.92, py + math.sin(ang) * L * 0.92)], fill=light + (255,), width=2)
    im = im.resize((S, S), Image.LANCZOS)
    bleed(im).save(path)


def conifer_atlas(path, palette, seed, S=1024):
    """2x2 cells of spruce/fir branch fronds (triangular, stem bottom->top)."""
    rng = np.random.default_rng(seed)
    K = 2 * S
    im = Image.new("RGBA", (K, K), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    C = K // 2
    for cell in range(4):
        ox, oy = (cell % 2) * C, (cell // 2) * C
        x0, y0, y1 = ox + C * 0.5, oy + C * 0.98, oy + C * 0.03
        # dense frond silhouette underneath: keeps alpha coverage at low mip levels (no vanishing at distance)
        sil = []
        for t in np.linspace(0, 1, 40):
            half = C * 0.42 * (1 - t) ** 0.85 + C * 0.04
            sil.append((x0 + half * (0.88 + 0.12 * rng.random()), y0 + (y1 - y0) * t - C * 0.06 * (1 - t)))
        sil += [(2 * x0 - px, py) for px, py in sil[::-1]]
        dr.polygon(sil, fill=(26, 46, 32, 255))
        dr.line([(x0, y0), (x0, y1)], fill=(80, 62, 44, 255), width=7)
        nb = 15
        for b in range(nb):
            t = (b + 0.5) / nb
            sy = y0 + (y1 - y0) * t
            for side in (-1, 1):
                L = C * 0.47 * (1 - t) ** 0.85 + C * 0.05
                a = -math.pi / 2 + side * rng.uniform(0.85, 1.15)
                ex, ey = x0 + math.cos(a) * L, sy + math.sin(a) * L
                dr.line([(x0, sy), (ex, ey)], fill=(78, 64, 46, 255), width=4)
                nn = int(L / 5)
                for k in range(nn):
                    u = k / max(nn, 1)
                    px, py = x0 + (ex - x0) * u, sy + (ey - sy) * u
                    for s2 in (-1, 1):
                        na = a + s2 * rng.uniform(0.5, 1.1) - 0.25
                        nl = rng.uniform(16, 26) * (1.0 - 0.4 * u)
                        col = _jit(palette[int(rng.integers(0, len(palette)))], rng, 9)
                        dr.line([(px, py), (px + math.cos(na) * nl, py + math.sin(na) * nl)], fill=col + (255,), width=4)
        # needles directly on main stem tip
        for k in range(60):
            py = y0 + (y1 - y0) * rng.uniform(0.6, 1.0)
            na = -math.pi / 2 + rng.uniform(-1.2, 1.2)
            col = _jit(palette[0], rng, 9)
            dr.line([(x0, py), (x0 + math.cos(na) * 22, py + math.sin(na) * 22)], fill=col + (255,), width=4)
    im = im.resize((S, S), Image.LANCZOS)
    bleed(im).save(path)


def fern_atlas(path, seed=6, S=1024):
    """2x2 cells of pinnate fern fronds (rachis bottom->top, leaflets both sides)."""
    rng = np.random.default_rng(seed)
    K = 2 * S; C = K // 2
    im = Image.new("RGBA", (K, K), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    pal = [(58, 104, 40), (70, 120, 46), (84, 132, 52), (50, 92, 36)]
    for cell in range(4):
        ox, oy = (cell % 2) * C, (cell // 2) * C
        x0, y0, y1 = ox + C * 0.5, oy + C * 0.99, oy + C * 0.03
        curve = rng.uniform(-0.08, 0.08) * C
        rach = lambda t: (x0 + curve * t * t, y0 + (y1 - y0) * t)
        dr.line([rach(t) for t in np.linspace(0, 1, 30)], fill=(70, 96, 40, 255), width=6)
        npin = 26
        for k in range(npin):
            t = 0.08 + 0.9 * k / npin
            px, py = rach(t)
            Lp = C * 0.36 * math.sin(math.pi * min(1, t * 1.1)) ** 0.8 + 6
            for side in (-1, 1):
                ang = -math.pi / 2 + side * rng.uniform(1.15, 1.35)
                col = _jit(pal[int(rng.integers(0, 4))], rng, 8)
                pl, pr = leaf_shape(px, py, ang, Lp, Lp * 0.13, tip=0.5)
                dr.polygon(pl + pr[::-1], fill=col + (255,))
                # lobes on each pinna
                for q in range(int(Lp / 14)):
                    u = (q + 0.5) / max(1, int(Lp / 14))
                    qx, qy = px + math.cos(ang) * Lp * u, py + math.sin(ang) * Lp * u
                    for s2 in (-1, 1):
                        la = ang + s2 * 1.2
                        ll = Lp * 0.1 * (1 - u * 0.6) + 4
                        dr.ellipse([qx + math.cos(la) * ll - ll * 0.55, qy + math.sin(la) * ll - ll * 0.55,
                                    qx + math.cos(la) * ll + ll * 0.55, qy + math.sin(la) * ll + ll * 0.55], fill=col + (255,))
    im = im.resize((S, S), Image.LANCZOS)
    bleed(im).save(path)


def grass_atlas(path, seed=5):
    """1024x512: top row = photo grass clumps (keyed from Poly Haven grass_medium_01),
    bottom row = 3 procedural lush grass clumps + 1 wildflower clump."""
    rng = np.random.default_rng(seed)
    src = np.asarray(Image.open(os.path.join(SRC, "models", "grass_medium_01", "textures", "grass_medium_01_diff_1k.jpg")).convert("RGB"), np.float32)
    strip = src[760:1016]
    L = strip.max(-1)
    a = np.clip((L - 20) / 28, 0, 1)
    hsv = np.asarray(Image.fromarray(strip.astype(np.uint8)).convert("HSV"), np.float32)
    hsv[..., 0] = (hsv[..., 0] + 7) % 255; hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255); hsv[..., 2] = np.clip(hsv[..., 2] * 1.35, 0, 255)
    strip = np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"), np.float32)
    top = np.concatenate([strip, a[..., None] * 255], -1)
    K = 512  # procedural cells drawn at 2x
    bot = Image.new("RGBA", (4 * K, K), (0, 0, 0, 0)); dr = ImageDraw.Draw(bot)
    for cell in range(4):
        ox = cell * K
        nbl = 150 if cell < 3 else 90
        for k in range(nbl):
            bx = ox + K * rng.normal(0.5, 0.14)
            h = K * rng.uniform(0.45, 0.97)
            lean = rng.normal(0, 0.25) * h
            w = rng.uniform(5, 9)
            dry = rng.random() < 0.12
            c0 = (92, 128, 46) if not dry else (140, 130, 78)
            c1 = (150, 196, 82) if not dry else (196, 182, 120)
            c0, c1 = _jit(c0, rng, 10), _jit(c1, rng, 12)
            segs = 6
            for s in range(segs):
                t0, t1 = s / segs, (s + 1) / segs
                def P(t, side):
                    return (bx + lean * t * t + side * w * (1 - t), K - 2 - h * t)
                col = tuple(int(c0[i] + (c1[i] - c0[i]) * t1) for i in range(3)) + (255,)
                dr.polygon([P(t0, -1), P(t0, 1), P(t1, 1), P(t1, -1)], fill=col)
        if cell == 3:   # wildflowers
            for k in range(26):
                fx, fy = ox + K * rng.uniform(0.15, 0.85), K * rng.uniform(0.15, 0.5)
                dr.line([(fx, fy), (fx + rng.normal(0, 8), K)], fill=(70, 100, 40, 255), width=4)
                col = [(245, 245, 235), (250, 210, 60), (170, 120, 210), (240, 240, 250)][int(rng.integers(0, 4))]
                r = rng.uniform(9, 15)
                for p in range(6):
                    a = p * math.pi / 3
                    dr.ellipse([fx + math.cos(a) * r - r * 0.6, fy + math.sin(a) * r - r * 0.6, fx + math.cos(a) * r + r * 0.6, fy + math.sin(a) * r + r * 0.6], fill=col + (255,))
                dr.ellipse([fx - r * 0.5, fy - r * 0.5, fx + r * 0.5, fy + r * 0.5], fill=(230, 170, 30, 255))
    bot = np.asarray(bot.resize((1024, 256), Image.LANCZOS), np.float32)
    atlas = np.concatenate([top, bot], 0)
    bleed(Image.fromarray(atlas.clip(0, 255).astype(np.uint8), "RGBA")).save(path)


# UV rects (u0,v0,u1,v1) image coords in grass atlas
GRASS_UV = [(205 / 1024, 10 / 512, 485 / 1024, 146 / 512), (600 / 1024, 15 / 512, 835 / 1024, 121 / 512),
            (30 / 1024, 125 / 512, 200 / 1024, 252 / 512), (245 / 1024, 140 / 512, 500 / 1024, 252 / 512),
            (590 / 1024, 125 / 512, 815 / 1024, 252 / 512),
            (0.0, 0.5, 0.25, 1.0), (0.25, 0.5, 0.5, 1.0), (0.5, 0.5, 0.75, 1.0)]
FLOWER_UV = (0.75, 0.5, 1.0, 1.0)


def birch_bark(path, S=512, seed=9):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (S, S), (222, 218, 206)); dr = ImageDraw.Draw(im)
    for k in range(260):
        y = rng.uniform(0, S); x = rng.uniform(0, S); w = rng.uniform(10, 70); h = rng.uniform(2, 6)
        c = int(rng.uniform(20, 60))
        dr.ellipse([x, y, x + w, y + h], fill=(c, c, c - 5))
    for k in range(400):
        y = rng.uniform(0, S); x = rng.uniform(0, S); w = rng.uniform(20, 120)
        c = int(rng.uniform(180, 235))
        dr.line([(x, y), (x + w, y)], fill=(c, c - 4, c - 14), width=1)
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    im.save(path)


def resized(src, dst, size):
    Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS).save(dst, quality=92)


def far_tree_atlas(path, seed=11, S=512):
    """Impostors for distant trees: 4 cells (2 conifer, 2 broadleaf), shaded silhouettes."""
    rng = np.random.default_rng(seed)
    K = 2 * S; C = K // 2
    im = Image.new("RGBA", (K, K), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    for cell in range(4):
        ox, oy = (cell % 2) * C, (cell // 2) * C
        conifer = cell < 2
        dr.rectangle([ox + C * 0.48, oy + C * 0.75, ox + C * 0.52, oy + C], fill=(60, 45, 32, 255))
        for k in range(900 if conifer else 700):
            if conifer:
                t = rng.uniform(0, 1); y = oy + C * (0.04 + 0.86 * t)
                half = C * 0.36 * t ** 0.9 + 4
                x = ox + C * 0.5 + rng.uniform(-half, half)
                base = (36, 66, 42)
            else:
                ang = rng.uniform(0, 2 * math.pi); rr = math.sqrt(rng.uniform(0, 1))
                x = ox + C * (0.5 + 0.4 * rr * math.cos(ang)); y = oy + C * (0.42 + 0.34 * rr * math.sin(ang))
                base = (58, 92, 44)
            shade = 0.65 + 0.6 * (1 - (y - oy) / C) * rng.uniform(0.7, 1.0)
            if not conifer:
                shade *= 0.8 + 0.4 * (0.5 - (x - ox) / C + 0.5)
            col = tuple(int(np.clip(c * shade + rng.normal(0, 6), 0, 255)) for c in base)
            r = rng.uniform(5, 13)
            dr.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))
    im = im.resize((S, S), Image.LANCZOS)
    bleed(im).save(path)


def canopy_texture(path, S=1024, seed=12):
    """Tileable top-down canopy texture for far hills (mottled tree crowns)."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (S, S), (22, 38, 24)); dr = ImageDraw.Draw(im)
    for k in range(5200):
        x, y = rng.uniform(0, S, 2); r = rng.uniform(6, 22)
        g = rng.uniform(0.55, 1.15)
        base = [(34, 60, 34), (44, 72, 36), (26, 48, 32), (52, 80, 40)][int(rng.integers(0, 4))]
        col = tuple(int(np.clip(c * g, 0, 255)) for c in base)
        for dx in (-S, 0, S):
            for dy in (-S, 0, S):
                dr.ellipse([x + dx - r, y + dy - r, x + dx + r, y + dy + r], fill=col)
                dr.ellipse([x + dx - r * 0.5 - r * 0.2, y + dy - r * 0.5 - r * 0.3, x + dx + r * 0.3, y + dy + r * 0.2], fill=tuple(min(255, int(c * 1.18)) for c in col))
    im.filter(ImageFilter.GaussianBlur(1.0)).save(path, quality=90)


def camo_texture(path, stops, seed, S=512, blobs=5):
    from gen_assets import tileable_noise, colorize
    n = tileable_noise(S, seed, 5)
    q = np.floor(n * blobs) / (blobs - 1)
    img = colorize(q, stops) * (0.92 + 0.16 * np.random.default_rng(seed).random((S, S)))[..., None]
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(path)
