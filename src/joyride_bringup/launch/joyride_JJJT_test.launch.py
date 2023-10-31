# -*- coding: utf-8 -*-


# A test launch file for the Joyride project Fall 2023
# Includes the following nodes:
# - CAN Server
# - Diagnostics (aggregator, automode manager, computer monitor)
# - Transforms
# - ROS Bagger
# - Cameras
# - Lidar 2D
# - Vectornav INS
# - Faked GPS
# - GPS-based Localization
# - Vel Preprocessor
# - Static Transform
# - Joystick
# - Preception
# - Navsat Transform
# - Simple Pose Controller
# - Navstack


# Python
from http.server import executable
import os
from ament_index_python.packages import get_package_share_directory

# ROS
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import (LifecycleNode, Node)
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument ,EmitEvent, RegisterEventHandler, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.events import matches_action
from launch_ros.event_handlers import OnStateTransition #, OnProcessStart
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from nav2_common.launch import RewrittenYaml



def generate_launch_description():

    default_diagnostic = os.path.join(get_package_share_directory('joyride_bringup'), 'config', 'diagnostic_minimal.yaml')

    diagnostic_config = LaunchConfiguration('diagnostic_config', default=default_diagnostic)

    diagnostic_param_cmd = DeclareLaunchArgument(
        'diagnostic_config',
        default_value=default_diagnostic,
        description='Path to diagnostic config file'
    )

    can_config = os.path.join(
        get_package_share_directory('joyride_ros2_socketcan'),
        'config',
        'can_msg_config.yaml'
    )

    bag_name = 'remote_control'

    obstacle_detect_config = os.path.join(
        get_package_share_directory('joyride_bringup'),
        'config',
        'obstacle_config.yaml'
    )

    bag_name = 'remote_control'

    controller_type = LaunchConfiguration('controller_type')
    nav_params = LaunchConfiguration('nav_params')

    declare_controller_type_cmd = DeclareLaunchArgument(
        'controller_type',
        default_value='purepursuit_config.yaml'
    )

    final_nav_params_path = DeclareLaunchArgument(
        'nav_params',
        default_value=[get_package_share_directory('joyride_bringup'), '/config/',LaunchConfiguration('controller_type')])




    return LaunchDescription([
        diagnostic_param_cmd,

        # ----------- ROSCAN Server ----------- #

        Node(
            package='joyride_ros2_socketcan',
            output='screen',
            namespace='servers',
            executable='roscan_server',
            name='roscan_interface',
            parameters=[can_config],
        ),

       # # Diagnostics
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('joyride_bringup'), 'launch'),
            '/joyride_diagnostics.launch.py'
            ]),
            launch_arguments={'diagnostic_config': diagnostic_config}.items()
        ),

        # ROSBAGGER
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('joyride_bringup'), 'launch'),
            '/rosbag_recorder.launch.py'
            ]),
            launch_arguments={'bag_name':'outdoor_navtest'}.items()
        ),

       # # Cameras
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('blackfly_camera_driver'), 'launch'),
                '/bfly_center_lifecycle.launch.py'
            ])
            ),

         # 2D Lidar
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('joyride_bringup'), 'launch'),
                '/lidar2D.launch.py'
        ])
        ),

       # # Vectornav
        #IncludeLaunchDescription(
        #    PythonLaunchDescriptionSource([os.path.join(
        #        get_package_share_directory('vectornav'), 'launch'),
        #        '/vectornav.launch.py'
        #])
        #),

       # # Faked Localization
        #Node(
        #    package='joyride_localization',
        #    executable='fake_odom',
        #    name='fake_odom',
        #    output='screen',
        #),

        # GPS-based Localization
        IncludeLaunchDescription(
             PythonLaunchDescriptionSource([os.path.join(
             get_package_share_directory('joyride_bringup'), 'launch'),
             '/gps_localization.launch.py'
             ])
        ),

       # # Velocity Preprocessor
        Node(
            package='joyride_control_py',
            executable='vel_preprocessor',
            name='vel_node',
        ),

       # # Static transforms
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('joyride_bringup'), 'launch'),
            '/joyride_transforms.launch.py'
            ])
        ),

       ## # Joystick
        #IncludeLaunchDescription(
        #    PythonLaunchDescriptionSource([os.path.join(
        #    get_package_share_directory('joyride_hmi'), 'launch'),
        #    '/joystick_bringup.launch.py'
        #])
        #),

       # # Perception
       # IncludeLaunchDescription(
       #     PythonLaunchDescriptionSource([os.path.join(
       #     get_package_share_directory('joyride_bringup'), 'launch'),
       #     '/joystick_perception.launch.py'
       # ])
       # ),

       # #Simple Pose Controller
       # Node(
       #     package="joyride_control_py",
       #     executable="simple_go_to_pose",
       #     name="go_to_pose",
       # )


        # Navstack
       # IncludeLaunchDescription(
       #     PythonLaunchDescriptionSource([os.path.join(
       #     get_package_share_directory('joyride_bringup'), 'launch'),
       #     '/navstack.launch.py']),
      #      launch_arguments={'params_file':nav_params}.items()
      #  ),



    ])