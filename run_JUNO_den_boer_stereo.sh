#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/thesis_orbslam/Examples/ROS
rosrun ORB_SLAM3 Stereo ./Vocabulary/ORBvoc.txt ./Examples/Stereo/Juno_den_boer.yaml false /stereo/left/image_raw:=/daheng_camera_manager/left/image_rect /stereo/right/image_raw:=/daheng_camera_manager/right/image_rect /stereo/left/camera_info:=/daheng_camera_manager/left/camera_info /stereo/right/camera_info:=/daheng_camera_manager/right/camera_info
