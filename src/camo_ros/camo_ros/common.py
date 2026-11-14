"""Shared helpers: image conversion, terrain lookup, pinhole camera ray casting."""
import os, math
import numpy as np
from sensor_msgs.msg import Image

CLASS_NAMES = ["person", "vehicle/tent", "animal"]   # YOLO ids 0,1,2  (gz labels 1,2,3)

WS = os.environ.get("CAMO_WS", "/mnt/c/ES-proj/camo_ws")
TERRAIN_SIZE = 240.0
TERRAIN_H = 14.0
CAM_OFFSET = np.array([0.15, 0.0, -0.02])
CAM_PITCH = 0.87           # rad, camera tilted down
CAM_HFOV = 1.25
IMG_W, IMG_H = 640, 480


def img_to_np(msg: Image) -> np.ndarray:
    a = np.frombuffer(msg.data, dtype=np.uint8)
    ch = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    a = a.reshape(msg.height, msg.width, ch) if ch == 3 else a.reshape(msg.height, msg.width)
    return a  # rgb8 from gazebo


def np_to_img(arr: np.ndarray, stamp=None, encoding="bgr8", frame="camera") -> Image:
    m = Image()
    if stamp is not None:
        m.header.stamp = stamp
    m.header.frame_id = frame
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.step = m.width * arr.shape[2]
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


class TerrainSampler:
    def __init__(self):
        import cv2
        p = os.path.join(WS, "models", "savanna_terrain", "textures", "heightmap.png")
        h = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        self.h = h.astype(np.float32) / 255.0 if h is not None else np.zeros((513, 513), np.float32)
        self.n = self.h.shape[0]

    def z(self, x, y):
        n = self.n
        u = min(max((x / TERRAIN_SIZE + 0.5) * (n - 1), 0), n - 1.001)
        v = min(max((0.5 - y / TERRAIN_SIZE) * (n - 1), 0), n - 1.001)
        i, j = int(v), int(u); fu, fv = u - j, v - i
        h = (self.h[i, j] * (1 - fu) * (1 - fv) + self.h[i, j + 1] * fu * (1 - fv)
             + self.h[i + 1, j] * (1 - fu) * fv + self.h[i + 1, j + 1] * fu * fv)
        return float(h * TERRAIN_H)


def quat_to_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_to_rpy(q):
    sinr = 2 * (q.w * q.x + q.y * q.z); cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = 2 * (q.w * q.y - q.z * q.x)
    return math.atan2(sinr, cosr), math.asin(max(-1, min(1, sinp))), quat_to_yaw(q)


def pixel_to_world(px, py, drone_pos, yaw, terrain: TerrainSampler):
    """Cast a ray through pixel (px,py) and intersect it with the terrain (drone assumed level)."""
    fx = (IMG_W / 2) / math.tan(CAM_HFOV / 2)
    # optical frame (x right, y down, z fwd) -> body frame (x fwd, y left, z up)
    x, y, z = 1.0, -(px - IMG_W / 2) / fx, -(py - IMG_H / 2) / fx
    cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)      # camera pitched nose-down
    xb = x * cp + z * sp
    zb = x * -sp + z * cp
    d_body = np.array([xb, y, zb])
    cy, sy = math.cos(yaw), math.sin(yaw)
    dw = np.array([cy * d_body[0] - sy * d_body[1], sy * d_body[0] + cy * d_body[1], d_body[2]])
    origin = np.array(drone_pos, float) + np.array([cy * CAM_OFFSET[0], sy * CAM_OFFSET[0], CAM_OFFSET[2]])
    if dw[2] >= -1e-3:
        return None
    t, step = 0.0, 0.5
    while t < 600:
        t += step; step *= 1.02
        p = origin + dw * t
        if p[2] <= terrain.z(p[0], p[1]):
            lo, hi = t - step, t
            for _ in range(12):
                mid = (lo + hi) / 2
                pm = origin + dw * mid
                if pm[2] <= terrain.z(pm[0], pm[1]): hi = mid
                else: lo = mid
            return origin + dw * hi
    return None
