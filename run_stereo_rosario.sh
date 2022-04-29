#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/thesis_orbslam/Examples/ROS
rosrun ORB_SLAM3 Stereo ./Vocabulary/ORBvoc.txt ./Examples/Stereo/Rosario.yaml false /stereo/left/image_raw:=/stereo/left/image_rect /stereo/right/image_raw:=/stereo/right/image_rect 
