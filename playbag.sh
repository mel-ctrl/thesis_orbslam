#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/meltem/thesis_orbslam/Examples/ROS

rosbag="/media/meltem/moo/Meltem/Thesis/Datasets/Own/mc0006_20220523_123903_road_rainy_corrected.bag"
mapping="/daheng_camera_manager/left/image_rect:=/daheng_camera_manager/left/image_raw /daheng_camera_manager/right/image_rect:=/daheng_camera_manager/right/image_raw"
rate="-r 0.8"
rosbag play $rosbag $mapping $rate

