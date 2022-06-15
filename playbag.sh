#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/thesis_orbslam/Examples/ROS

#rosbag="/media/meltem/moo/Meltem/Thesis/Datasets/Own/mc0006_20220523_123903_road_rainy_corrected.bag"
rosbag="/media/meltem/moo/Meltem/Thesis/Datasets/Own/juno_normal.bag"
mapping="/daheng_camera_manager/left/image_rect:=/stereo/left/image_raw /daheng_camera_manager/right/image_rect:=/stereo/right/image_raw /daheng_camera_manager/left/camera_info:=/stereo/left/camera_info /daheng_camera_manager/right/camera_info:=/stereo/right/camera_info"
rate=--rate=0.8
rosbag play $rosbag $mapping $rate

