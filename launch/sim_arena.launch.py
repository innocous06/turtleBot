import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot_pe')
    world_file = os.path.join(pkg_share, 'worlds', 'arena_10x10.sdf')

    ros_ign_gazebo_share = get_package_share_directory('ros_ign_gazebo')
    ign_gazebo_launch = os.path.join(ros_ign_gazebo_share, 'launch', 'ign_gazebo.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=world_file,
            description='Path to SDF world file'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ign_gazebo_launch),
            launch_arguments={'ign_args': [LaunchConfiguration('world'), ' -r']}.items(),
        )
    ])
