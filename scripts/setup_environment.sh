#!/bin/bash
set -e

echo "=== TurtleBot P&E Environment Setup ==="

# 1. Update and install basic dependencies
sudo apt update && sudo apt install -y software-properties-common curl wget lsb-release gnupg

# 2. ROS2 Humble setup (if not already installed)
if ! command -v ros2 &> /dev/null; then
    echo "Installing ROS2 Humble..."
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install -y ros-humble-desktop ros-dev-tools python3-colcon-common-extensions
fi

# 3. Ignition Fortress repository & installation
echo "Installing Ignition Fortress..."
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install -y ignition-fortress

# 4. TurtleBot 4 simulator & ros_ign bridge
echo "Installing TurtleBot 4 Simulation & Bridge packages..."
sudo apt install -y \
  ros-humble-turtlebot4-simulator \
  ros-humble-turtlebot4-description \
  ros-humble-turtlebot4-msgs \
  ros-humble-turtlebot4-navigation \
  ros-humble-turtlebot4-node \
  ros-humble-turtlebot4-desktop \
  ros-humble-irobot-create-nodes \
  ros-humble-ros-ign-bridge \
  ros-humble-ros-ign-gazebo \
  ros-humble-ros-ign-interfaces

# 5. Python dependencies
pip3 install numpy scipy pytest

echo "=== Setup Complete ==="
echo "To build and run in your ROS2 workspace:"
echo "  mkdir -p ~/turtlebot_pe_ws/src"
echo "  cp -r $(pwd) ~/turtlebot_pe_ws/src/"
echo "  cd ~/turtlebot_pe_ws"
echo "  source /opt/ros/humble/setup.bash"
echo "  colcon build --packages-select turtlebot_pe"
echo "  source install/setup.bash"
echo "  ros2 launch turtlebot_pe catcher.launch.py"
