#!/usr/bin/env python3
"""Save one frame from an image topic -> argv[1]  (needs ros_gz_bridge running).
usage: snap.py out.png [topic] [skip_frames]"""
import sys, rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
rclpy.init(); n = Node("snap"); out = sys.argv[1]; got = []
topic = sys.argv[2] if len(sys.argv) > 2 else "/camo/camera/image"
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
def cb(m): got.append(CvBridge().imgmsg_to_cv2(m, "bgr8"))
n.create_subscription(Image, topic, cb, 1)
t = n.get_clock().now()
while rclpy.ok() and len(got) <= skip and (n.get_clock().now() - t).nanoseconds < 120e9:
    rclpy.spin_once(n, timeout_sec=0.5)
if got: cv2.imwrite(out, got[-1]); print("saved", out, got[-1].shape)
else: print("NO FRAME")
