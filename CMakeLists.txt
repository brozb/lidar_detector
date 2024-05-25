cmake_minimum_required(VERSION 3.0.2)
project(lidar_detector)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  tf2_ros
  tf2_py
  tf
  ros_numpy
  pcl_ros
  grid_map_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidar_detector
  CATKIN_DEPENDS geometry_msgs rospy std_msgs tf2_ros tf2_py ros_numpy pcl_ros grid_map_msgs
#  DEPENDS system_lib
)

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/detect_ground.py
  scripts/detect_leader.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(lidar_filter src/lidar_filter.cpp)
target_link_libraries(lidar_filter ${catkin_LIBRARIES})
