#!/usr/bin/env python3
"""Auto-label dataset collector (YOLO format) using Gazebo's bounding-box camera.

Teleports the drone around ground-truth targets (via /world/savanna/set_pose),
saves RGB frames + ground-truth boxes. Run once per generated world (different --seed).
ros2 run camo_ros dataset_node --ros-args -p n_samples:=300 -p tag:=w42
"""
import os, json, math, random
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from .common import img_to_np, WS, TerrainSampler, CAM_PITCH, IMG_W, IMG_H, WORLD_NAME


class DatasetNode(Node):
    def __init__(self):
        super().__init__("camo_dataset")
        self.declare_parameter("n_samples", 300)
        self.declare_parameter("tag", "w0")
        self.declare_parameter("neg_fraction", 0.15)
        self.n = self.get_parameter("n_samples").value
        self.tag = self.get_parameter("tag").value
        self.neg = self.get_parameter("neg_fraction").value
        self.out = os.path.join(WS, "dataset")
        for s in ("train", "val"):
            os.makedirs(f"{self.out}/images/{s}", exist_ok=True); os.makedirs(f"{self.out}/labels/{s}", exist_ok=True)
        self.targets = json.load(open(os.path.join(WS, "worlds", "targets.json")))["targets"]
        self.terrain = TerrainSampler()
        self.img, self.boxes, self.img_stamp = None, None, 0
        self.create_subscription(Image, "/camo/camera/image", self.on_img, 1)
        self.create_subscription(Detection2DArray, "/camo/camera/boxes", self.on_box, 1)
        self.cli = self.create_client(SetEntityPose, f"/world/{WORLD_NAME}/set_pose")
        self.i, self.state, self.wait_frames, self.fut = 0, "INIT", 0, None
        self.saved_boxes = 0
        self.create_timer(0.1, self.step)

    def on_img(self, m):
        self.img = img_to_np(m).copy(); self.img_stamp += 1

    def on_box(self, m):
        self.boxes = m

    def teleport(self):
        if random.random() < self.neg or not self.targets:
            tx, ty = random.uniform(-80, 80), random.uniform(-80, 80)
        else:
            t = random.choice(self.targets); tx, ty = t["x"], t["y"]
        alt = random.uniform(22, 36)
        yaw = random.uniform(-math.pi, math.pi)
        d = alt / math.tan(CAM_PITCH) * random.uniform(0.8, 1.25)
        lat = random.uniform(-4, 4)
        x = tx - d * math.cos(yaw) - lat * math.sin(yaw); y = ty - d * math.sin(yaw) + lat * math.cos(yaw)
        x, y = max(-100, min(100, x)), max(-100, min(100, y))
        z = max(self.terrain.z(x + a, y + b) for a in (-6, 0, 6) for b in (-6, 0, 6)) + alt
        req = SetEntityPose.Request()
        req.entity.name = "camo_drone"; req.entity.type = Entity.MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, z
        req.pose.orientation.z = math.sin(yaw / 2); req.pose.orientation.w = math.cos(yaw / 2)
        self.fut = self.cli.call_async(req)

    def step(self):
        if self.state == "INIT":
            if self.cli.service_is_ready() and self.img is not None:
                self.state = "MOVE"
            return
        if self.state == "MOVE":
            self.boxes = None; self.teleport(); self.state = "WAITSRV"
        elif self.state == "WAITSRV":
            if self.fut.done():
                self.mark = self.img_stamp; self.state = "SETTLE"
        elif self.state == "SETTLE":
            if self.img_stamp - self.mark >= 6 and self.boxes is not None:
                self.save(); self.i += 1
                if self.i >= self.n:
                    self.get_logger().info(f"DONE: {self.i} frames, {self.saved_boxes} boxes"); rclpy.shutdown(); return
                self.state = "MOVE"
            elif self.img_stamp - self.mark > 60:
                self.state = "MOVE"

    def save(self):
        split = "val" if random.random() < 0.15 else "train"
        name = f"{self.tag}_{self.i:05d}"
        lines = []
        for d in self.boxes.detections:
            try:
                lab = int(float(d.results[0].hypothesis.class_id)) if d.results else 0
            except ValueError:
                continue
            if lab not in (1, 2, 3):
                continue
            cx, cy, w, h = d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y
            x1, y1, x2, y2 = max(0, cx - w / 2), max(0, cy - h / 2), min(IMG_W, cx + w / 2), min(IMG_H, cy + h / 2)
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue
            if (x2 - x1) > 0.9 * IMG_W and (y2 - y1) > 0.9 * IMG_H:   # bogus full-frame artefact
                continue
            lines.append(f"{lab - 1} {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} {(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}")
        cv2.imwrite(f"{self.out}/images/{split}/{name}.jpg", cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{self.out}/labels/{split}/{name}.txt", "w").write("\n".join(lines))
        self.saved_boxes += len(lines)
        if self.i % 25 == 0:
            self.get_logger().info(f"{self.i}/{self.n} frames, {self.saved_boxes} boxes")


def main():
    rclpy.init(); n = DatasetNode()
    try:
        rclpy.spin(n)
    except Exception:
        pass
