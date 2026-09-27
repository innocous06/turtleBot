import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot_pe')
    default_config_path = os.path.join(pkg_share, 'config', 'runner_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_config_path,
            description='Path to ROS2 YAML parameters file for Runner'
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='Topic for 2D LaserScan'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Topic for Odometry'
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Topic for velocity commands'
        ),
        Node(
            package='turtlebot_pe',
            executable='runner_node',
            name='runner_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'scan_topic': LaunchConfiguration('scan_topic'),
                    'odom_topic': LaunchConfiguration('odom_topic'),
                    'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                }
            ],
            remappings=[
                ('/scan', LaunchConfiguration('scan_topic')),
                ('/odom', LaunchConfiguration('odom_topic')),
                ('/cmd_vel', LaunchConfiguration('cmd_vel_topic')),
            ]
        )
    ])
