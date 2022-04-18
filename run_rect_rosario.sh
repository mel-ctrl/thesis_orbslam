#!/bin/bash
source /opt/ros/noetic/setup.bash
cd /home/meltem/catkin_ws
source devel/setup.bash

ROS_NAMESPACE=stereo rosrun stereo_image_proc stereo_image_proc
