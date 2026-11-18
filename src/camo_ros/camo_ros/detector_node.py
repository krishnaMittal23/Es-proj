#!/usr/bin/env python3
"""Camouflage detector node.

sub : /camo/camera/image                (sensor_msgs/Image, rgb8)
pub : /camo/detections                  (vision_msgs/Detection2DArray)
      /camo/image_annotated             (sensor_msgs/Image, bgr8)

mode:=yolo      fine-tuned YOLOv8 (trained on Gazebo data)
mode:=baseline  classical colour-anomaly detector (shows why camouflage is hard)
"""
import os, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from .common import img_to_np, np_to_img, CLASS_NAMES, WS

COLORS = [(0, 255, 255), (255, 160, 0), (0, 255, 0)]  # BGR per class


class BaselineDetector:
    """Local colour-contrast detector: flag blobs that differ from their surroundings."""

    def __call__(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg = cv2.GaussianBlur(lab, (0, 0), 25)
        dev = np.sqrt(((lab - bg) ** 2).sum(axis=2))
        dev = cv2.GaussianBlur(dev, (0, 0), 2)
        mask = (dev > 18).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if 120 < a < 20000 and 0.2 < w / max(h, 1) < 5:
                out.append((x, y, x + w, y + h, min(0.99, float(dev[y:y + h, x:x + w].mean()) / 60), 0))
        return out


class YoloDetector:
    def __init__(self, weights, conf, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return out


class DetectorNode(Node):
    def __init__(self):
        super().__init__("camo_detector")
        self.declare_parameter("mode", "yolo")
        self.declare_parameter("weights", os.path.join(WS, "runs", "camo_yolo", "weights", "best.pt"))
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("imgsz", 800)
        mode = self.get_parameter("mode").value
        self.mode = mode
        if mode == "yolo":
            w = self.get_parameter("weights").value
            if not os.path.exists(w):
                self.get_logger().warn(f"weights {w} missing -> falling back to baseline")
                self.mode = "baseline"
        self.det = (YoloDetector(self.get_parameter("weights").value, self.get_parameter("conf").value,
                                 self.get_parameter("imgsz").value) if self.mode == "yolo" else BaselineDetector())
        self.get_logger().info(f"detector mode: {self.mode}")
        self.pub_d = self.create_publisher(Detection2DArray, "/camo/detections", 5)
        self.pub_i = self.create_publisher(Image, "/camo/image_annotated", 5)
        self.create_subscription(Image, "/camo/camera/image", self.cb, 1)
        self.fps, self.t_last = 0.0, time.time()

    def cb(self, msg):
        rgb = img_to_np(msg)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dets = self.det(bgr)
        arr = Detection2DArray(); arr.header = msg.header
        vis = bgr.copy()
        for x1, y1, x2, y2, c, k in dets:
            d = Detection2D(); d.header = msg.header
            d.bbox.center.position.x = (x1 + x2) / 2; d.bbox.center.position.y = (y1 + y2) / 2
            d.bbox.size_x = float(x2 - x1); d.bbox.size_y = float(y2 - y1)
            h = ObjectHypothesisWithPose(); h.hypothesis.class_id = str(k); h.hypothesis.score = c
            d.results.append(h); arr.detections.append(d)
            col = COLORS[k % 3]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, f"{CLASS_NAMES[k % 3]} {c:.2f}", (int(x1), max(12, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        now = time.time(); self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_last, 1e-3); self.t_last = now
        cv2.putText(vis, f"{self.mode.upper()}  {self.fps:4.1f} fps  det={len(dets)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.pub_d.publish(arr)
        self.pub_i.publish(np_to_img(vis, msg.header.stamp))


def main():
    rclpy.init(); rclpy.spin(DetectorNode()); rclpy.shutdown()


if __name__ == "__main__":
    main()
