cmake_minimum_required(VERSION 3.8)
project(video_object_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(video_reader_node src/video_reader_node.cpp)
ament_target_dependencies(video_reader_node
  rclcpp sensor_msgs cv_bridge image_transport OpenCV
)

add_executable(tracker_node src/tracker_node.cpp)
ament_target_dependencies(tracker_node
  rclcpp std_msgs sensor_msgs visualization_msgs 
  geometry_msgs cv_bridge OpenCV tf2 tf2_ros
)

install(TARGETS
  video_reader_node
  tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
