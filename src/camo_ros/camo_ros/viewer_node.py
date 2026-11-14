#!/usr/bin/env python3
"""Demo window: annotated drone camera + mission status HUD (OpenCV, shows through WSLg)."""
import cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .common import img_to_np


class Viewer(Node):
    def __init__(self):
        super().__init__("camo_viewer")
        self.img, self.status = None, "waiting..."
        self.create_subscription(Image, "/camo/image_annotated", lambda m: setattr(self, "img", img_to_np(m)), 1)
        self.create_subscription(String, "/camo/status", lambda m: setattr(self, "status", m.data), 1)
        self.create_timer(1 / 30, self.draw)
        cv2.namedWindow("Camouflage Detection - Drone Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camouflage Detection - Drone Camera", 960, 720)
        try:
            cv2.setWindowProperty("Camouflage Detection - Drone Camera", cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def draw(self):
        if self.img is None:
            f = np.zeros((480, 640, 3), np.uint8)
        else:
            f = self.img.copy()
        cv2.rectangle(f, (0, f.shape[0] - 30), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
        cv2.putText(f, self.status, (8, f.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.imshow("Camouflage Detection - Drone Camera", f)
        cv2.waitKey(1)


def main():
    rclpy.init(); rclpy.spin(Viewer()); rclpy.shutdown()
