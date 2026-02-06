from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('video_path', default_value=''),
        DeclareLaunchArgument('fps', default_value='25.0'),
        
        Node(
            package='video_object_tracker',
            executable='video_reader_node',
            name='video_reader',
            output='screen',
            parameters=[{
                'video_path': LaunchConfiguration('video_path'),
                'fps': LaunchConfiguration('fps'),
                'loop': True
            }]
        ),
        
        Node(
            package='video_object_tracker',
            executable='tracker_node',
            name='tracker',
            output='screen',
            parameters=[{
                'iou_threshold': 0.3,
                'trajectory_length': 100,
                'max_missing': 30,
                'min_area': 500.0
            }]
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),
    ])
