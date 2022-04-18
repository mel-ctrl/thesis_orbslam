#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/test/Examples/ROS
rosrun ORB_SLAM3 Stereo ./Vocabulary/ORBvoc.txt ./Examples/Stereo/Rosario.yaml false /camera/left/image_raw:=/stereo/left/image_rect /camera/right/image_raw:=/stereo/right/image_rect
