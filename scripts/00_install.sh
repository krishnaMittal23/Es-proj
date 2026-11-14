#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release software-properties-common git wget unzip \
  python3-pip python3-venv mesa-utils vulkan-tools x11-apps build-essential
sudo add-apt-repository -y universe
# ROS2 apt source
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/ros2.list
# Gazebo apt source
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt-get update
sudo apt-get install -y ros-jazzy-desktop ros-jazzy-ros-gz gz-harmonic \
  ros-jazzy-cv-bridge ros-dev-tools python3-colcon-common-extensions python3-opencv python3-numpy
grep -q "jazzy/setup.bash" ~/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo INSTALL_DONE
