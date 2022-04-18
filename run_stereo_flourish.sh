#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/test/Examples/ROS
rosrun ORB_SLAM3 Stereo ./Vocabulary/ORBvoc.txt ./Examples/Stereo/Flourish.yaml false /camera/left/image_raw:=sensor/camera/vi_sensor/left/image_rect /camera/right/image_raw:=sensor/camera/vi_sensor/right/image_rect
