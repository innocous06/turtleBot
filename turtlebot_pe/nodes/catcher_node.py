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
from turtlebot_pe.core.catcher_controller import CatcherController


class CatcherNode(Node):
    def __init__(self):
        super().__init__("catcher_node")

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
        self.declare_parameter("ekf_lost_timeout", 3.0)

        # Detector
        self.declare_parameter("bg_learning_duration", 2.0)
        self.declare_parameter("foreground_threshold", 0.15)
        self.declare_parameter("min_cluster_size", 1)
        self.declare_parameter("max_cluster_size", 30)
        self.declare_parameter("cluster_distance_threshold", 0.30)
        self.declare_parameter("min_target_width", 0.04)
        self.declare_parameter("max_target_width", 0.60)
        self.declare_parameter("target_type", "any")

        # PN & Herding
        self.declare_parameter("pn_gain_N", 3.0)
        self.declare_parameter("pn_bearing_gain", 1.8)
        self.declare_parameter("pn_speed_coupling", 0.5)
        self.declare_parameter("herd_offset", 0.75)
        self.declare_parameter("herd_corner_offset", 1.50)
        self.declare_parameter("corner_wall_threshold", 2.0)

        # Shadowing
        self.declare_parameter("shadow_enter_distance", 0.70)
        self.declare_parameter("shadow_exit_distance", 0.80)
        self.capture_radius = 0.50
        self.declare_parameter("capture_radius", self.capture_radius)
        self.declare_parameter("capture_hold_target", 1.0)
        self.declare_parameter("shadow_trail_distance", 0.35)
        self.declare_parameter("shadow_Kx", 2.5)
        self.declare_parameter("shadow_Ky", 4.0)
        self.declare_parameter("shadow_Ktheta", 3.0)

        # DWA & Search
        self.declare_parameter("dwa_prediction_time", 1.5)
        self.declare_parameter("dwa_safety_margin", 0.10)
        self.declare_parameter("search_spin_rate", 1.0)
        self.declare_parameter("search_timeout", 12.6)

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

        self.controller = CatcherController(
            v_max=v_max,
            omega_max=omega_max,
            robot_radius=robot_radius,
            arena_bounds=arena_bounds,
            pn_gain_N=float(self.get_parameter("pn_gain_N").value),
            pn_bearing_gain=float(self.get_parameter("pn_bearing_gain").value),
            pn_speed_coupling=float(self.get_parameter("pn_speed_coupling").value),
            herd_offset=float(self.get_parameter("herd_offset").value),
            herd_corner_offset=float(self.get_parameter("herd_corner_offset").value),
            corner_wall_threshold=float(self.get_parameter("corner_wall_threshold").value),
            shadow_enter_distance=float(self.get_parameter("shadow_enter_distance").value),
            shadow_exit_distance=float(self.get_parameter("shadow_exit_distance").value),
            capture_radius=float(self.get_parameter("capture_radius").value),
            capture_hold_target=float(self.get_parameter("capture_hold_target").value),
            shadow_trail_distance=float(self.get_parameter("shadow_trail_distance").value),
            shadow_Kx=float(self.get_parameter("shadow_Kx").value),
            shadow_Ky=float(self.get_parameter("shadow_Ky").value),
            shadow_Ktheta=float(self.get_parameter("shadow_Ktheta").value),
            dwa_prediction_time=float(self.get_parameter("dwa_prediction_time").value),
            dwa_safety_margin=float(self.get_parameter("dwa_safety_margin").value),
            search_spin_rate=float(self.get_parameter("search_spin_rate").value),
            search_timeout=float(self.get_parameter("search_timeout").value),
            ekf_lost_timeout=float(self.get_parameter("ekf_lost_timeout").value),
        )

        self.ego_pose = (0.0, 0.0, 0.0)
        self.ego_vel = (0.0, 0.0)
        self.has_odom = False

        self.latest_scan: Optional[LaserScan] = None
        self.last_detection_time = 0.0
        self.start_sim_time: Optional[float] = None
        self.current_sim_time = 0.0

        if "rclpy" in sys.modules:
            self.sub_scan = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
            self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
            self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 10)
            self.pub_target_estimate = self.create_publisher(PoseStamped, "/catcher/target_estimate", 10)
            self.pub_state = self.create_publisher(String, "/catcher/state", 10)
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
                    if accepted:
                        self.last_detection_time = self.current_sim_time
                else:
                    self.ekf.handle_occlusion()
            else:
                self.ekf.handle_occlusion()

        time_since_detection = self.current_sim_time - self.last_detection_time
        target_state = self.ekf.get_state() if self.ekf.is_initialized() else None

        v_cmd, omega_cmd, state_str = self.controller.compute_control(
            ego_pose=self.ego_pose,
            ego_vel=self.ego_vel,
            target_ekf_state=target_state,
            obstacles=self.obstacle_map.get_obstacles(),
            dt=self.dt,
            bg_ready=self.detector.is_background_ready(),
            time_since_target_detected=time_since_detection,
        )

        cmd_msg = Twist()
        cmd_msg.linear.x = float(v_cmd)
        cmd_msg.angular.z = float(omega_cmd)
        if "rclpy" in sys.modules:
            self.pub_cmd_vel.publish(cmd_msg)

            state_msg = String()
            state_msg.data = f"{state_str}:{self.controller.get_capture_timer():.2f}"
            self.pub_state.publish(state_msg)

            if target_state is not None:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.latest_scan.header.stamp if self.latest_scan else self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "odom"
                pose_msg.pose.position.x = float(target_state[0])
                pose_msg.pose.position.y = float(target_state[1])
                self.pub_target_estimate.publish(pose_msg)

            if self.controller.is_capture_confirmed():
                capture_elapsed = self.current_sim_time - (self.start_sim_time or 0.0)
                self.get_logger().info(f"[CAPTURE CONFIRMED] Time to capture: {capture_elapsed:.2f}s")


def main(args=None):
    if "rclpy" not in sys.modules:
        return
    rclpy.init(args=args)
    node = CatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
