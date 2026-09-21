# turtlebot_pe

Autonomous Catcher navigation and pursuit package for the TurtleBot Pursuit & Evasion Challenge (Round 1 Catcher Qualification).

## System Requirements

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Ignition Gazebo Fortress
- TurtleBot 4 Simulation Stack (`turtlebot4_simulator`, `irobot_create_nodes`)
- Python 3.10+ with `numpy`, `pytest`

## Implemented Architecture

The package implements a modular pursuit pipeline split between pure Python control logic and a ROS 2 node wrapper:

- **Perception (`LidarTargetDetector`)**: Background subtraction against static arena walls/obstacles and Euclidean clustering. Supports targets down to 0.04m width (moving sphere/dot mode) up to full-sized TurtleBots.
- **State Estimation (`TargetEKF`)**: 5-state CTRV (Coordinated Turn Rate and Velocity) Extended Kalman Filter with analytical Jacobians, Mahalanobis gating (5.991 threshold), and covariance inflation during obstacle occlusion.
- **Mapping (`ObstacleMap`)**: Runtime 2D occupancy grid built from initial background scans with Bresenham line-of-sight raycasting.
- **Decision & Control (`CatcherController`)**: 
  - Finite State Machine: `INIT` -> `SEARCH` -> `PURSUE` -> `SHADOW` -> `CAPTURED`.
  - Augmented Proportional Navigation (PN) with center-directed wall/corner herding.
  - Dynamic Window Approach (DWA) circular arc collision avoidance against static obstacles and arena perimeter.
  - Lyapunov-stable trailing controller for velocity-matched target shadowing.

## Interfaces & Running Variables

### Subscribed Topics (Inputs)

| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/scan` | `sensor_msgs/msg/LaserScan` | 2D LIDAR range scan (360 deg) |
| `/odom` | `nav_msgs/msg/Odometry` | Catcher pose and twist in odom frame |

### Published Topics (Outputs)

| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Commanded linear (`linear.x`) and angular (`angular.z`) velocities |
| `/catcher/state` | `std_msgs/msg/String` | State machine status and capture hold duration (`STATE:SECONDS`) |
| `/catcher/target_estimate` | `geometry_msgs/msg/PoseStamped` | Filtered target world coordinates from EKF |

### Key Parameters (`config/catcher_params.yaml`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `scan_topic` | string | `"/scan"` | LIDAR topic name |
| `odom_topic` | string | `"/odom"` | Odometry topic name |
| `cmd_vel_topic` | string | `"/cmd_vel"` | Velocity command topic name |
| `v_max` | float | `0.30` | Maximum linear velocity limit (m/s) |
| `omega_max` | float | `1.80` | Maximum angular velocity limit (rad/s) |
| `robot_radius` | float | `0.18` | Collision radius of the robot (m) |
| `arena_min_x`, `arena_max_x` | float | `-5.0`, `5.0` | Arena X boundaries (10m span) |
| `arena_min_y`, `arena_max_y` | float | `-5.0`, `5.0` | Arena Y boundaries (10m span) |
| `target_type` | string | `"any"` | Target filter mode: `"any"` (sphere/dot/robot) or `"robot"` |
| `min_target_width` | float | `0.04` | Minimum cluster diameter for valid target detection (m) |
| `pn_gain_N` | float | `3.0` | Proportional navigation navigation constant |
| `herd_offset` | float | `0.75` | Center-bias herding offset ahead of target (m) |
| `capture_radius` | float | `0.50` | Official capture threshold distance (m) |
| `capture_hold_target` | float | `1.0` | Required continuous hold duration inside capture radius (s) |
| `shadow_trail_distance` | float | `0.35` | Trailing distance behind target during shadowing (m) |

## Build & Execution

### 1. Build Package
```bash
mkdir -p ~/turtlebot_pe_ws/src
cp -r /path/to/turtlebot_pe ~/turtlebot_pe_ws/src/
cd ~/turtlebot_pe_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select turtlebot_pe
source install/setup.bash
```

### 2. Run Test Suite
Unit and integration tests run without ROS 2 runtime dependencies:
```bash
pytest test/
```

### 3. Launch Simulation Arena
```bash
ros2 launch turtlebot_pe sim_arena.launch.py
```

### 4. Launch Autonomous Catcher
```bash
ros2 launch turtlebot_pe catcher.launch.py
```

Topic namespacing can be overridden via launch arguments:
```bash
ros2 launch turtlebot_pe catcher.launch.py scan_topic:=/catcher/scan odom_topic:=/catcher/odom cmd_vel_topic:=/catcher/cmd_vel
```

## Repository Structure

```
turtlebot_pe/
├── config/
│   └── catcher_params.yaml
├── launch/
│   ├── catcher.launch.py
│   └── sim_arena.launch.py
├── models/
│   └── obstacle_box/
├── resource/
├── scripts/
│   ├── run_benchmark.sh
│   └── setup_environment.sh
├── test/
│   ├── test_catcher_controller.py
│   ├── test_ekf.py
│   ├── test_extreme_grill.py
│   ├── test_integration.py
│   ├── test_lidar_detector.py
│   └── test_rigorous_edge_cases.py
├── turtlebot_pe/
│   ├── core/
│   │   ├── catcher_controller.py
│   │   ├── ekf_tracker.py
│   │   ├── lidar_detector.py
│   │   ├── math_utils.py
│   │   └── obstacle_map.py
│   ├── nodes/
│   │   └── catcher_node.py
│   └── viz/
│       └── marker_publisher.py
├── worlds/
│   └── arena_10x10.sdf
├── package.xml
├── README.md
├── setup.cfg
└── setup.py
```
