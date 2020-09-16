cmake_minimum_required(VERSION 2.8.3)
SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")

project(ur_control)

set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH} $ENV{QT5_INSTALL_PATH})

find_package(catkin REQUIRED COMPONENTS
  actionlib
  control_msgs
  roscpp
  sensor_msgs
  trac_ik_lib
  trajectory_msgs
  rosbag
        message_generation
#        eigen_conversions
#        eigen_stl_containers
#        eigen_typekit
)


catkin_package(

  CATKIN_DEPENDS

)



include_directories(include
  ${catkin_INCLUDE_DIRS}
)

#find_package(Eigen REQUIRED)
#include_directories(${Eigen_INCLUDE_DIR})

add_executable(ur_control src/catch_bottle.cpp src/catch_bottle.h)
target_link_libraries(ur_control ${catkin_LIBRARIES})
