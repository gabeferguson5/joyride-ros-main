# Python
from http.server import executable
import os
from ament_index_python.packages import get_package_share_directory

# ROS
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import (DeclareLaunchArgument, EmitEvent, RegisterEventHandler)
from launch_ros.actions import Node, LifecycleNode
from launch.events import matches_action

from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch.event_handlers import OnProcessStart
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
#from launch.action import DeclareLaunchArgument
#from launch.substitutions import TextSubstitution


def generate_launch_description():
    
    roscan_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/can_server_bringup.launch.py'
        ])
    )

    vectornav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('vectornav'), 'launch'),
        '/vectornav.launch.py'
        ])
    )


    rosbagger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/data_log_bringup.launch.py'
        ])
    )

    blackfly_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('joyride_core'), 'launch'),
        '/blackfly_bringup.launch.py'
        ])
    )


    return LaunchDescription([
        roscan_server_launch,
        vectornav_launch,
        rosbagger_launch,
        blackfly_launch


    ])