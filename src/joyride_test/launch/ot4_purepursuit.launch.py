
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    bag_name = 'purepursuit_test'

    return LaunchDescription([
        
        # Bagging
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('joyride_bringup'), 'launch'),
            '/data_log_bringup.launch.py'
            ]),
            launch_arguments={'bag_name':bag_name}.items()
        ),

        # Navigator and dependencies
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('joyride_bringup'), 'launch'),
            '/navigation.launch.py']),
            
        )
    ])
