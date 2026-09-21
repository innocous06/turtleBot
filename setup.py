import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'turtlebot_pe'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dharm',
    maintainer_email='dharm@example.com',
    description='Autonomous Pursuit and Evasion Challenge algorithms for TurtleBot 4 Lite',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'catcher_node = turtlebot_pe.nodes.catcher_node:main',
        ],
    },
)
