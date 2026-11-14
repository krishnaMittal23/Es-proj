#!/usr/bin/env python3
"""Save one frame from /camo/camera/image -> argv[1]  (needs ros_gz_bridge running)."""
import sys, rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
rclpy.init(); n = Node("snap"); out = sys.argv[1]; got = []
def cb(m): got.append(CvBridge().imgmsg_to_cv2(m, "bgr8"))
n.create_subscription(Image, "/camo/camera/image", cb, 1)
t = n.get_clock().now()
while rclpy.ok() and not got and (n.get_clock().now() - t).nanoseconds < 60e9:
    rclpy.spin_once(n, timeout_sec=0.5)
if got: cv2.imwrite(out, got[-1]); print("saved", out, got[-1].shape)
else: print("NO FRAME")
