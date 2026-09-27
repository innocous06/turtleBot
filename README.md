# turtlebot_pe

Autonomous pursuit-evasion package for the TurtleBot Pursuit & Evasion Challenge, covering both Round 1 (Catcher Qualification) and Round 2 (1v1 Knockout).

## System Requirements

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Ignition Gazebo Fortress
- TurtleBot 4 Simulation Stack (`turtlebot4_simulator`, `irobot_create_nodes`)
- Python 3.10+ with `numpy`, `pytest`

## Implemented Architecture

The package implements a modular pipeline with decoupled pure-Python algorithms and thin ROS 2 lifecycle node wrappers:

- **Perception (`LidarTargetDetector`)**: Background subtraction against static arena walls/obstacles and Euclidean clustering. Supports targets down to 0.04m width (moving sphere/dot mode) up to standard TurtleBots (0.15m–0.60m).
- **State Estimation (`TargetEKF`)**: 5-state CTRV (Coordinated Turn Rate and Velocity) Extended Kalman Filter with analytical Jacobians, Mahalanobis gating (5.991 threshold), and covariance inflation during obstacle occlusion.
- **Mapping (`ObstacleMap`)**: Runtime 2D occupancy grid built from initial background scans with Bresenham line-of-sight raycasting.
- **Catcher Controller (`CatcherController`)**: 
  - Finite State Machine: `INIT` -> `SEARCH` -> `PURSUE` -> `SHADOW` -> `CAPTURED`.
  - Augmented Proportional Navigation (PN) with center-directed wall/corner herding.
  - Dynamic Window Approach (DWA) circular arc collision avoidance against static obstacles and arena perimeter.
  - Lyapunov-stable trailing controller for velocity-matched target shadowing (0.5m / 1.0s hold).
- **Runner Controller (`RunnerController`)**:
  - Finite State Machine: `INIT` -> `EVADE` -> `SHIELD` -> `SURVIVED`.
  - Artificial Potential Fields (APF) with dynamic closing-velocity scaling.
  - Tangential obstacle vortex fields preventing local minima and corner traps.
  - Dynamic obstacle shielding (orbiting behind static obstacles to break pursuer line-of-sight).
  - Anti-prediction Ornstein-Uhlenbeck noise for randomized heading perturbation when pursuer is within 2.0m.
  - End-game mode ($t \ge 150\text{s}$) maximizing final distance from the Catcher for tie-breaking.

## Interfaces & Running Variables

### Subscribed Topics (Inputs)

| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/scan` (or `/catcher/scan`, `/runner/scan`) | `sensor_msgs/msg/LaserScan` | 2D LIDAR range scan (360 deg) |
| `/odom` (or `/catcher/odom`, `/runner/odom`) | `nav_msgs/msg/Odometry` | Robot pose and twist in odom frame |

### Published Topics (Outputs)

| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Commanded linear (`linear.x`) and angular (`angular.z`) velocities |
| `/catcher/state` | `std_msgs/msg/String` | Catcher FSM status and capture hold duration (`STATE:SECONDS`) |
| `/runner/state` | `std_msgs/msg/String` | Runner FSM status and match elapsed time (`STATE:SECONDS`) |
| `/catcher/target_estimate` | `geometry_msgs/msg/PoseStamped` | Filtered target position estimated by Catcher |
| `/runner/target_estimate` | `geometry_msgs/msg/PoseStamped` | Filtered pursuer position estimated by Runner |

### Key Configuration Parameters

#### Catcher Parameters (`config/catcher_params.yaml`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `v_max` | `0.30` | Maximum linear velocity limit (m/s) |
| `omega_max` | `1.80` | Maximum angular velocity limit (rad/s) |
| `target_type` | `"any"` | Target filter mode: `"any"` (sphere/dot/robot) or `"robot"` |
| `min_target_width` | `0.04` | Minimum cluster diameter for valid target detection (m) |
| `pn_gain_N` | `3.0` | Proportional navigation navigation constant |
| `herd_offset` | `0.75` | Center-bias herding offset ahead of target (m) |
| `capture_radius` | `0.50` | Official capture threshold distance (m) |
| `capture_hold_target` | `1.0` | Required continuous hold duration inside capture radius (s) |
| `shadow_trail_distance` | `0.35` | Trailing distance behind target during shadowing (m) |

#### Runner Parameters (`config/runner_params.yaml`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `v_max` | `0.30` | Maximum linear velocity limit (m/s) |
| `omega_max` | `1.80` | Maximum angular velocity limit (rad/s) |
| `apf_eta_pursuer` | `3.5` | Repulsion gain away from Catcher |
| `apf_beta_closing` | `1.5` | Closing velocity multiplier for pursuer repulsion |
| `apf_k_vortex` | `1.2` | Tangential vortex field gain around obstacles |
| `apf_eta_wall` | `2.5` | Repulsion gain from arena perimeter walls |
| `ou_sigma` | `0.5` | Anti-prediction heading noise intensity |
| `shield_activation_distance` | `2.0` | Distance to Catcher triggering obstacle shield check (m) |
| `endgame_start` | `150.0` | Time threshold to switch to max-distance tie-breaker mode (s) |

## Build & Execution

### 1. Build Package
```bash
mkdir -p ~/turtlebot_pe_ws/src
cp -r /path/to/turtleBot ~/turtlebot_pe_ws/src/turtlebot_pe
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

### 4. Round 1: Launch Autonomous Catcher
```bash
ros2 launch turtlebot_pe catcher.launch.py
```

### 5. Round 2: Launch Autonomous Runner
```bash
ros2 launch turtlebot_pe runner.launch.py
```

### 6. Round 2: Launch Full 1v1 Match Simulation
Launches arena, spawns Catcher at (-1.5, 0) and Runner at (1.5, 0), and executes both nodes simultaneously:
```bash
ros2 launch turtlebot_pe full_match.launch.py
```

## Repository Structure

```
turtleBot/
├── config/
│   ├── catcher_params.yaml
│   └── runner_params.yaml
├── launch/
│   ├── catcher.launch.py
│   ├── full_match.launch.py
│   ├── runner.launch.py
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
│   ├── test_rigorous_edge_cases.py
│   ├── test_rigorous_simulation.py
│   ├── test_runner_controller.py
│   └── test_tactical_edge_cases.py
├── turtlebot_pe/
│   ├── core/
│   │   ├── catcher_controller.py
│   │   ├── ekf_tracker.py
│   │   ├── lidar_detector.py
│   │   ├── math_utils.py
│   │   ├── obstacle_map.py
│   │   └── runner_controller.py
│   ├── nodes/
│   │   ├── catcher_node.py
│   │   └── runner_node.py
│   └── viz/
│       └── marker_publisher.py
├── worlds/
│   └── arena_10x10.sdf
├── package.xml
├── README.md
├── setup.cfg
└── setup.py
```
