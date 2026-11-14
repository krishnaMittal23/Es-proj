"""One-command demo:  ros2 launch camo_ros demo.launch.py  [mode:=yolo|baseline] [gui:=true] [rviz:=false]"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WS = os.environ.get("CAMO_WS", "/mnt/c/ES-proj/camo_ws")

BRIDGE = [
    "/camo/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
    "/camo/camera/boxes@vision_msgs/msg/Detection2DArray[gz.msgs.AnnotatedAxisAligned2DBox_V",
    "/camo/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
    "/camo/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
    "/camo/image_annotated@sensor_msgs/msg/Image]gz.msgs.Image",
    "/world/savanna/set_pose@ros_gz_interfaces/srv/SetEntityPose",
]


def generate_launch_description():
    a = lambda n, d: DeclareLaunchArgument(n, default_value=d)
    world = os.path.join(WS, "worlds", "savanna.sdf")
    return LaunchDescription([
        a("mode", "yolo"), a("gui", "true"), a("rviz", "false"), a("detector", "true"),
        a("autostart", "true"), a("viewer", "true"), a("conf", "0.35"),
        ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen", condition=IfCondition(LaunchConfiguration("gui"))),
        ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "--headless-rendering", world], output="screen",
                       condition=UnlessCondition(LaunchConfiguration("gui"))),
        TimerAction(period=10.0, actions=[
            ExecuteProcess(cmd=["bash", os.path.join(WS, "scripts", "follow_cam.sh")], output="screen",
                           condition=IfCondition(LaunchConfiguration("gui")))]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", arguments=BRIDGE, output="log"),
        TimerAction(period=8.0, actions=[
            Node(package="camo_ros", executable="detector_node", output="screen",
                 parameters=[{"mode": LaunchConfiguration("mode"), "conf": LaunchConfiguration("conf")}],
                 condition=IfCondition(LaunchConfiguration("detector"))),
            Node(package="camo_ros", executable="mission_node", output="screen",
                 parameters=[{"autostart": LaunchConfiguration("autostart")}]),
            Node(package="camo_ros", executable="viewer_node", output="log", condition=IfCondition(LaunchConfiguration("viewer"))),
            ExecuteProcess(cmd=["rviz2", "-d", os.path.join(WS, "config", "camo.rviz")], output="log",
                           condition=IfCondition(LaunchConfiguration("rviz"))),
        ]),
    ])
