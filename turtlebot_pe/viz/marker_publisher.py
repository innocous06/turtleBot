from typing import Any

try:
    from visualization_msgs.msg import Marker
except ImportError:
    Marker = Any


class TargetMarkerPublisher:
    @staticmethod
    def create_target_marker(x: float, y: float, frame_id: str = "odom"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = "target_estimate"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.15
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.30
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.8
        return marker

    @staticmethod
    def create_capture_zone_marker(x: float, y: float, capture_radius: float = 0.50, frame_id: str = "odom"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = "capture_zone"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.02
        marker.scale.x = float(capture_radius * 2.0)
        marker.scale.y = float(capture_radius * 2.0)
        marker.scale.z = 0.04
        marker.color.r = 0.2
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = 0.3
        return marker
