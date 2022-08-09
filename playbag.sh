#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/thesis_orbslam/Examples/ROS

#rosbag="/media/meltem/moo/Own/den_boer_mc0038_20220613_095019_sunny.bag"
rosbag="/media/meltem/moo/Own/den_boer_mc0038_20220613_095019_sunny.bag"
mapping="/stereo/left/image_raw:=/daheng_camera_manager/left/image_rect /stereo/right/image_raw:=/daheng_camera_manager/right/image_rect /stereo/left/camera_info:=/daheng_camera_manager/left/camera_info /stereo/right/camera_info:=/daheng_camera_manager/right/camera_info"
rate=--rate=1.0
rosbag play $rosbag $mapping $rate

