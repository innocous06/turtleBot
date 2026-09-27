import sys
import math
from typing import Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist, PoseStamped
    from std_msgs.msg import String
except ImportError:
    class Node:
        def __init__(self, *args, **kwargs):
            pass

from turtlebot_pe.core.math_utils import wrap_angle
from turtlebot_pe.core.ekf_tracker import TargetEKF
from turtlebot_pe.core.lidar_detector import LidarTargetDetector
from turtlebot_pe.core.obstacle_map import ObstacleMap
from turtlebot_pe.core.runner_controller import RunnerController


class RunnerNode(Node):
    def __init__(self):
        super().__init__("runner_node")

        # Topics
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # Motion & bounds
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("v_max", 0.30)
        self.declare_parameter("omega_max", 1.80)
        self.declare_parameter("robot_radius", 0.18)
        self.declare_parameter("arena_min_x", -5.0)
        self.declare_parameter("arena_min_y", -5.0)
        self.declare_parameter("arena_max_x", 5.0)
        self.declare_parameter("arena_max_y", 5.0)

        # EKF
        self.declare_parameter("ekf_dt", 0.05)
        self.declare_parameter("ekf_sigma_a", 0.5)
        self.declare_parameter("ekf_sigma_alpha", 1.2)
        self.declare_parameter("ekf_sigma_r", 0.03)
        self.declare_parameter("ekf_sigma_phi", 0.02)

        # Detector
        self.declare_parameter("bg_learning_duration", 2.0)
        self.declare_parameter("foreground_threshold", 0.15)
        self.declare_parameter("min_cluster_size", 3)
        self.declare_parameter("max_cluster_size", 30)
        self.declare_parameter("cluster_distance_threshold", 0.30)
        self.declare_parameter("min_target_width", 0.15)
        self.declare_parameter("max_target_width", 0.60)
        self.declare_parameter("target_type", "robot")

        # APF
        self.declare_parameter("apf_eta_pursuer", 3.5)
        self.declare_parameter("apf_d_pursuer_max", 4.0)
        self.declare_parameter("apf_beta_closing", 1.5)
        self.declare_parameter("apf_eta_obstacle", 2.0)
        self.declare_parameter("apf_d_obstacle_max", 0.80)
        self.declare_parameter("apf_k_vortex", 1.2)
        self.declare_parameter("apf_eta_wall", 2.5)
        self.declare_parameter("apf_d_wall_max", 1.0)
        self.declare_parameter("apf_k_center", 0.08)
        self.declare_parameter("heading_gain", 2.5)

        # Anti-prediction
        self.declare_parameter("ou_tau", 0.8)
        self.declare_parameter("ou_sigma", 0.5)
        self.declare_parameter("ou_activation_distance", 2.0)

        # Shielding
        self.declare_parameter("shield_activation_distance", 2.0)
        self.declare_parameter("shield_time_advantage", 0.5)
        self.declare_parameter("shield_orbit_margin", 0.35)
        self.declare_parameter("shield_exit_distance", 3.0)

        # Timing
        self.declare_parameter("match_duration", 180.0)
        self.declare_parameter("endgame_start", 150.0)
        self.declare_parameter("endgame_distance_gain", 0.5)
        self.declare_parameter("endgame_ou_sigma", 0.2)

        scan_topic = self.get_parameter("scan_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.dt = 1.0 / rate_hz

        v_max = float(self.get_parameter("v_max").value)
        omega_max = float(self.get_parameter("omega_max").value)
        robot_radius = float(self.get_parameter("robot_radius").value)
        arena_bounds = (
            float(self.get_parameter("arena_min_x").value),
            float(self.get_parameter("arena_min_y").value),
            float(self.get_parameter("arena_max_x").value),
            float(self.get_parameter("arena_max_y").value),
        )

        self.ekf = TargetEKF(
            dt=self.dt,
            sigma_a=float(self.get_parameter("ekf_sigma_a").value),
            sigma_alpha=float(self.get_parameter("ekf_sigma_alpha").value),
            sigma_r=float(self.get_parameter("ekf_sigma_r").value),
            sigma_phi=float(self.get_parameter("ekf_sigma_phi").value),
        )

        self.detector = LidarTargetDetector(
            bg_learning_duration=float(self.get_parameter("bg_learning_duration").value),
            foreground_threshold=float(self.get_parameter("foreground_threshold").value),
            min_cluster_size=int(self.get_parameter("min_cluster_size").value),
            max_cluster_size=int(self.get_parameter("max_cluster_size").value),
            cluster_distance_threshold=float(self.get_parameter("cluster_distance_threshold").value),
            min_target_width=float(self.get_parameter("min_target_width").value),
            max_target_width=float(self.get_parameter("max_target_width").value),
            target_type=str(self.get_parameter("target_type").value),
        )

        self.obstacle_map = ObstacleMap(
            resolution=0.10,
            size=12.0,
            arena_bounds=arena_bounds,
        )

        self.controller = RunnerController(
            v_max=v_max,
            omega_max=omega_max,
            robot_radius=robot_radius,
            arena_bounds=arena_bounds,
            dt=self.dt,
            apf_eta_pursuer=float(self.get_parameter("apf_eta_pursuer").value),
            apf_d_pursuer_max=float(self.get_parameter("apf_d_pursuer_max").value),
            apf_beta_closing=float(self.get_parameter("apf_beta_closing").value),
            apf_eta_obstacle=float(self.get_parameter("apf_eta_obstacle").value),
            apf_d_obstacle_max=float(self.get_parameter("apf_d_obstacle_max").value),
            apf_k_vortex=float(self.get_parameter("apf_k_vortex").value),
            apf_eta_wall=float(self.get_parameter("apf_eta_wall").value),
            apf_d_wall_max=float(self.get_parameter("apf_d_wall_max").value),
            apf_k_center=float(self.get_parameter("apf_k_center").value),
            heading_gain=float(self.get_parameter("heading_gain").value),
            ou_tau=float(self.get_parameter("ou_tau").value),
            ou_sigma=float(self.get_parameter("ou_sigma").value),
            ou_activation_distance=float(self.get_parameter("ou_activation_distance").value),
            shield_activation_distance=float(self.get_parameter("shield_activation_distance").value),
            shield_time_advantage=float(self.get_parameter("shield_time_advantage").value),
            shield_orbit_margin=float(self.get_parameter("shield_orbit_margin").value),
            shield_exit_distance=float(self.get_parameter("shield_exit_distance").value),
            match_duration=float(self.get_parameter("match_duration").value),
            endgame_start=float(self.get_parameter("endgame_start").value),
            endgame_distance_gain=float(self.get_parameter("endgame_distance_gain").value),
            endgame_ou_sigma=float(self.get_parameter("endgame_ou_sigma").value),
        )

        self.ego_pose = (0.0, 0.0, 0.0)
        self.ego_vel = (0.0, 0.0)
        self.has_odom = False

        self.latest_scan: Optional[LaserScan] = None
        self.start_sim_time: Optional[float] = None
        self.current_sim_time = 0.0

        if "rclpy" in sys.modules:
            self.sub_scan = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
            self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
            self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 10)
            self.pub_target_estimate = self.create_publisher(PoseStamped, "/runner/target_estimate", 10)
            self.pub_state = self.create_publisher(String, "/runner/state", 10)
            self.timer = self.create_timer(self.dt, self.control_loop)

    def odom_callback(self, msg: Odometry) -> None:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.ego_pose = (float(px), float(py), float(yaw))
        self.ego_vel = (float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))
        self.has_odom = True

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.current_sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start_sim_time is None:
            self.start_sim_time = self.current_sim_time

        if not self.detector.is_background_ready() and self.has_odom:
            self.obstacle_map.add_scan(
                ranges=list(msg.ranges),
                angle_min=float(msg.angle_min),
                angle_increment=float(msg.angle_increment),
                ego_pose=self.ego_pose,
            )

    def control_loop(self) -> None:
        if not self.has_odom:
            return

        self.ekf.predict()

        if self.detector.is_background_ready() and not self.obstacle_map._is_finalized:
            self.obstacle_map.finalize()

        if self.latest_scan is not None:
            if self.ekf.is_initialized():
                target_pos = self.ekf.get_position()
                xp, yp, th_p = self.ego_pose
                dx = target_pos[0] - xp
                dy = target_pos[1] - yp
                r_pred = math.hypot(dx, dy)
                phi_pred = wrap_angle(math.atan2(dy, dx) - th_p)
                self.detector.set_ekf_prediction(r_pred, phi_pred)

            detection = self.detector.process_scan(
                ranges=list(self.latest_scan.ranges),
                angle_min=float(self.latest_scan.angle_min),
                angle_increment=float(self.latest_scan.angle_increment),
                timestamp=self.current_sim_time,
                range_min=float(self.latest_scan.range_min),
                range_max=float(self.latest_scan.range_max),
            )

            if detection is not None:
                r_meas, phi_meas = detection
                abs_phi = self.ego_pose[2] + phi_meas
                meas_x = self.ego_pose[0] + r_meas * math.cos(abs_phi)
                meas_y = self.ego_pose[1] + r_meas * math.sin(abs_phi)

                has_los = True
                if self.obstacle_map._is_finalized:
                    has_los = self.obstacle_map.has_line_of_sight(
                        (self.ego_pose[0], self.ego_pose[1]), (meas_x, meas_y)
                    )

                if has_los:
                    accepted, _ = self.ekf.update([r_meas, phi_meas], self.ego_pose)
                else:
                    self.ekf.handle_occlusion()
            else:
                self.ekf.handle_occlusion()

        elapsed_time = self.current_sim_time - (self.start_sim_time or self.current_sim_time)
        target_state = self.ekf.get_state() if self.ekf.is_initialized() else None

        v_cmd, omega_cmd, state_str = self.controller.compute_control(
            ego_pose=self.ego_pose,
            catcher_ekf_state=target_state,
            obstacles=self.obstacle_map.get_obstacles(),
            elapsed_time=elapsed_time,
            dt=self.dt,
            bg_ready=self.detector.is_background_ready(),
        )

        cmd_msg = Twist()
        cmd_msg.linear.x = float(v_cmd)
        cmd_msg.angular.z = float(omega_cmd)

        if "rclpy" in sys.modules:
            self.pub_cmd_vel.publish(cmd_msg)

            state_msg = String()
            state_msg.data = f"{state_str}:{elapsed_time:.1f}s"
            self.pub_state.publish(state_msg)

            if target_state is not None:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.latest_scan.header.stamp if self.latest_scan else self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "odom"
                pose_msg.pose.position.x = float(target_state[0])
                pose_msg.pose.position.y = float(target_state[1])
                self.pub_target_estimate.publish(pose_msg)

            if state_str == "SURVIVED":
                self.get_logger().info(f"[MATCH SURVIVED] Completed full 180s duration at distance: {float(np.linalg.norm(np.array(self.ego_pose[:2]) - target_state[:2])) if target_state is not None else 0.0:.2f}m")


def main(args=None):
    if "rclpy" not in sys.modules:
        return
    rclpy.init(args=args)
    node = RunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
