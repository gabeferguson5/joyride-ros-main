

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from nav2_common.launch import RewrittenYaml


def generate_launch_description():

    params_path = os.path.join(get_package_share_directory('joyride_core'), 'config', 'nav_config.yaml')

    # -------------------- Common -------------------- #

    roscan_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/can_server_bringup.launch.py'
        ])
    )

    static_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/static_transforms.launch.py'
        ])
    )

    # -------------------- Localization -------------------- #

    odom_ackermann = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/odom_bringup.launch.py'
    ])
    )


    # -------------------- Nav 2 -------------------- #

    navstack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/joyride_navstack.launch.py']),
        launch_arguments={
            'params_file': params_path,
        }.items()
    )


    return LaunchDescription([
        
        Node(
            package='joyride_control',
            executable='vel_preprocess_node',
            name='vel_node',
        ),

        static_tf_launch,
        roscan_server_launch,

        #localizer_launch,
        odom_ackermann,

        navstack_launch
    ])
