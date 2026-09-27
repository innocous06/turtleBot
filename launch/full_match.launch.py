import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot_pe')
    world_file = os.path.join(pkg_share, 'worlds', 'arena_10x10.sdf')
    catcher_config = os.path.join(pkg_share, 'config', 'catcher_params.yaml')
    runner_config = os.path.join(pkg_share, 'config', 'runner_params.yaml')

    sim_arena_launch = os.path.join(pkg_share, 'launch', 'sim_arena.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=world_file,
            description='Path to SDF world file'
        ),
        # Launch simulation arena
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_arena_launch),
            launch_arguments={'world': LaunchConfiguration('world')}.items(),
        ),
        # Catcher Node (under /catcher namespace)
        Node(
            package='turtlebot_pe',
            executable='catcher_node',
            name='catcher_node',
            namespace='catcher',
            output='screen',
            parameters=[
                catcher_config,
                {
                    'scan_topic': '/catcher/scan',
                    'odom_topic': '/catcher/odom',
                    'cmd_vel_topic': '/catcher/cmd_vel',
                }
            ],
            remappings=[
                ('/scan', '/catcher/scan'),
                ('/odom', '/catcher/odom'),
                ('/cmd_vel', '/catcher/cmd_vel'),
            ]
        ),
        # Runner Node (under /runner namespace)
        Node(
            package='turtlebot_pe',
            executable='runner_node',
            name='runner_node',
            namespace='runner',
            output='screen',
            parameters=[
                runner_config,
                {
                    'scan_topic': '/runner/scan',
                    'odom_topic': '/runner/odom',
                    'cmd_vel_topic': '/runner/cmd_vel',
                }
            ],
            remappings=[
                ('/scan', '/runner/scan'),
                ('/odom', '/runner/odom'),
                ('/cmd_vel', '/runner/cmd_vel'),
            ]
        ),
    ])
