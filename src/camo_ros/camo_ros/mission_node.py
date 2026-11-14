#!/usr/bin/env python3
"""Mission controller: lawnmower search -> confirm -> geolocate -> report.

sub : /camo/odom (Odometry), /camo/detections (Detection2DArray)
pub : /camo/cmd_vel (Twist, body frame), /camo/status (String),
      /camo/found_markers (MarkerArray, for RViz), /camo/path (Path)
"""
import os, json, math, time
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from .common import TerrainSampler, quat_to_yaw, pixel_to_world, CLASS_NAMES, WS


class MissionNode(Node):
    def __init__(self):
        super().__init__("camo_mission")
        P = self.declare_parameter
        P("altitude_agl", 20.0)       # metres above ground
        P("speed", 6.0)
        P("lane_spacing", 20.0)
        P("area", 75.0)               # search square half-size (m)
        P("min_conf", 0.45)
        P("confirm_frames", 4)
        P("dedupe_radius", 5.0)
        P("match_radius", 6.0)        # for scoring vs ground truth
        P("autostart", True)
        g = lambda k: self.get_parameter(k).value
        self.alt, self.speed, self.spacing, self.area = g("altitude_agl"), g("speed"), g("lane_spacing"), g("area")
        self.min_conf, self.confirm_n = g("min_conf"), g("confirm_frames")
        self.dedupe_r, self.match_r = g("dedupe_radius"), g("match_radius")
        self.terrain = TerrainSampler()
        self.pos, self.yaw = None, 0.0
        self.wps = self._lawnmower()
        self.wp_i = 0
        self.state = "WAIT" if not g("autostart") else "TAKEOFF"
        self.found = []          # dict(x,y,z,cls,conf,t)
        self.tracks = []         # candidate tracks [x,y,count,cls,conf]
        self.hover_until = 0.0
        self.t0 = None
        self.dist = 0.0
        self.last_pos = None
        self.start_xy = None
        self.path = Path(); self.path.header.frame_id = "world"
        self.create_subscription(Odometry, "/camo/odom", self.on_odom, 10)
        self.create_subscription(Detection2DArray, "/camo/detections", self.on_det, 10)
        self.pub_cmd = self.create_publisher(Twist, "/camo/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "/camo/status", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "/camo/found_markers", 10)
        self.pub_path = self.create_publisher(Path, "/camo/path", 1)
        self.create_timer(0.05, self.loop)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info(f"mission ready: {len(self.wps)} waypoints, alt {self.alt} m AGL")

    # ------------------------------------------------------------- planning
    def _lawnmower(self):
        a, w = self.area, []
        xs = np.arange(-a, a + 0.1, self.spacing)
        for i, x in enumerate(xs):
            ys = (-a, a) if i % 2 == 0 else (a, -a)
            w += [(x, ys[0]), (x, ys[1])]
        return w

    def target_z(self, x, y):
        # fly at constant AGL w.r.t. max terrain near the point (look-ahead) -> never hits hills
        zmax = max(self.terrain.z(x + dx, y + dy) for dx in (-8, 0, 8) for dy in (-8, 0, 8))
        return zmax + self.alt

    # ------------------------------------------------------------- callbacks
    def on_odom(self, m):
        p = m.pose.pose.position
        self.pos = np.array([p.x, p.y, p.z]); self.yaw = quat_to_yaw(m.pose.pose.orientation)
        if self.start_xy is None:
            self.start_xy = (p.x, p.y)
        if self.last_pos is not None:
            self.dist += float(np.linalg.norm(self.pos[:2] - self.last_pos[:2]))
        self.last_pos = self.pos.copy()
        ps = PoseStamped(); ps.header.frame_id = "world"; ps.pose = m.pose.pose
        if len(self.path.poses) == 0 or np.linalg.norm(self.pos - np.array([self.path.poses[-1].pose.position.x, self.path.poses[-1].pose.position.y, self.path.poses[-1].pose.position.z])) > 1.0:
            self.path.poses.append(ps); self.path.header.stamp = self.get_clock().now().to_msg(); self.pub_path.publish(self.path)

    def on_det(self, m):
        if self.pos is None or self.state not in ("SEARCH", "CONFIRM"):
            return
        seen = set()
        for d in m.detections:
            sc = d.results[0].hypothesis.score
            if sc < self.min_conf:
                continue
            k = int(d.results[0].hypothesis.class_id)
            # use bottom-centre of the box (where object meets the ground)
            px = d.bbox.center.position.x; py = d.bbox.center.position.y + d.bbox.size_y * 0.35
            w = pixel_to_world(px, py, self.pos, self.yaw, self.terrain)
            if w is None or np.linalg.norm(w[:2] - self.pos[:2]) > 55 or np.max(np.abs(w[:2])) > 95:
                continue
            if any(np.linalg.norm(w[:2] - np.array([f["x"], f["y"]])) < self.dedupe_r for f in self.found):
                continue
            # associate with a candidate track
            for ti, t in enumerate(self.tracks):
                if ti not in seen and np.linalg.norm(w[:2] - t["p"]) < 4.0:
                    t["p"] = 0.7 * t["p"] + 0.3 * w[:2]; t["n"] += 1; t["conf"] = max(t["conf"], sc); t["last"] = time.time()
                    t["cls"] = k; seen.add(ti)
                    if t["n"] >= self.confirm_n:
                        self.confirm(t, w[2]); self.tracks = [x for x in self.tracks if x is not t]
                    break
            else:
                self.tracks.append(dict(p=w[:2].copy(), n=1, cls=k, conf=sc, last=time.time()))
        self.tracks = [t for t in self.tracks if time.time() - t["last"] < 2.0]

    def confirm(self, t, z):
        f = dict(x=float(t["p"][0]), y=float(t["p"][1]), z=float(z), cls=int(t["cls"]), conf=float(t["conf"]),
                 t=time.time() - self.t0)
        self.found.append(f)
        self.get_logger().info(f"*** TARGET #{len(self.found)}: {CLASS_NAMES[f['cls'] % 3]} at ({f['x']:.1f}, {f['y']:.1f}) conf {f['conf']:.2f}")
        self.state = "CONFIRM"; self.hover_until = time.time() + 1.5
        self.publish_markers()

    # ------------------------------------------------------------- control
    def cmd(self, vx, vy, vz, wz):
        t = Twist(); t.linear.x, t.linear.y, t.linear.z, t.angular.z = float(vx), float(vy), float(vz), float(wz)
        self.pub_cmd.publish(t)

    def fly_to(self, wx, wy, wz, speed, face=True):
        dx, dy, dz = wx - self.pos[0], wy - self.pos[1], wz - self.pos[2]
        dist = math.hypot(dx, dy)
        if dist > 0.5:
            want = math.atan2(dy, dx)
            err = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
            wz_cmd = max(-1.0, min(1.0, 1.5 * err)) if face else 0.0
        else:
            wz_cmd, err = 0.0, 0.0
        v = min(speed, 0.8 * dist + 0.5) * max(0.0, math.cos(err)) ** 2 if face else min(speed, dist)
        vz = max(-2.0, min(2.0, 1.2 * dz))
        self.cmd(v, 0.0, vz, wz_cmd)       # body-frame: x forward
        return dist, abs(dz)

    def loop(self):
        if self.pos is None:
            return
        if self.t0 is None:
            self.t0 = time.time()
        if self.state == "TAKEOFF":
            tz = self.target_z(self.pos[0], self.pos[1])
            self.cmd(0, 0, max(-2, min(2, 1.0 * (tz - self.pos[2]))), 0)
            if abs(tz - self.pos[2]) < 1.0:
                self.state = "SEARCH"; self.get_logger().info("airborne -> SEARCH")
        elif self.state in ("SEARCH", "CONFIRM"):
            if self.state == "CONFIRM":
                self.cmd(0, 0, 0, 0)
                if time.time() > self.hover_until:
                    self.state = "SEARCH"
                return
            wx, wy = self.wps[self.wp_i]
            d, _ = self.fly_to(wx, wy, self.target_z(self.pos[0], self.pos[1]), self.speed)
            if d < 3.0:
                self.wp_i += 1
                if self.wp_i >= len(self.wps):
                    self.state = "RTL"; self.get_logger().info("search complete -> RTL")
        elif self.state == "RTL":
            sx, sy = self.start_xy
            d, _ = self.fly_to(sx, sy, self.target_z(sx, sy), self.speed)
            if d < 3.0:
                self.state = "DONE"; self.cmd(0, 0, 0, 0); self.report()
        elif self.state == "DONE":
            self.cmd(0, 0, 0, 0)

    # ------------------------------------------------------------- reporting
    def publish_status(self):
        if self.t0 is None:
            return
        s = f"{self.state} | wp {min(self.wp_i, len(self.wps))}/{len(self.wps)} | found {len(self.found)} | t {time.time() - self.t0:4.0f}s | alt {self.pos[2]:.0f}m"
        self.pub_status.publish(String(data=s))
        self.publish_markers()

    def publish_markers(self):
        ma = MarkerArray()
        for i, f in enumerate(self.found):
            m = Marker(); m.header.frame_id = "world"; m.ns = "found"; m.id = i; m.type = Marker.SPHERE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = f["x"], f["y"], f["z"] + 1.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 3.0
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 0.8
            ma.markers.append(m)
        self.pub_mark.publish(ma)

    def report(self):
        gt_path = os.path.join(WS, "worlds", "targets.json")
        res = dict(found=self.found, duration_s=time.time() - self.t0, distance_m=self.dist)
        if os.path.exists(gt_path):
            gt = json.load(open(gt_path))["targets"]
            used, tp = set(), 0
            per = {}
            for g in gt:
                best, bi = 1e9, None
                for i, f in enumerate(self.found):
                    d = math.hypot(f["x"] - g["x"], f["y"] - g["y"])
                    if d < best and i not in used:
                        best, bi = d, i
                hit = bi is not None and best < self.match_r
                if hit:
                    used.add(bi); tp += 1
                key = "camouflaged" if g["camouflaged"] else "plain/animal"
                a = per.setdefault(key, [0, 0]); a[1] += 1; a[0] += int(hit)
                g["detected"] = hit
            fp = len(self.found) - tp
            res.update(ground_truth=gt, true_positives=tp, false_positives=fp,
                       recall=tp / max(len(gt), 1), precision=tp / max(len(self.found), 1), per_group=per)
            lines = ["", "=" * 54, " MISSION REPORT", "=" * 54,
                     f" targets in world : {len(gt)}", f" detected (TP)    : {tp}", f" false alarms (FP): {fp}",
                     f" recall           : {res['recall']:.0%}", f" precision        : {res['precision']:.0%}"]
            for k, (h, n) in per.items():
                lines.append(f"   {k:14s}: {h}/{n}")
            lines += [f" flight time      : {res['duration_s']:.0f}s   distance {self.dist:.0f} m", "=" * 54]
            for l in lines:
                self.get_logger().info(l)
        os.makedirs(os.path.join(WS, "results"), exist_ok=True)
        with open(os.path.join(WS, "results", "mission_report.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        self.pub_status.publish(String(data="DONE | report saved to results/mission_report.json"))


def main():
    rclpy.init(); rclpy.spin(MissionNode()); rclpy.shutdown()


if __name__ == "__main__":
    main()
