#!/bin/bash
source /opt/ros/noetic/setup.bash
source /home/meltem/catkin_ws/devel/setup.bash

ROS_NAMESPACE=stereo rosrun stereo_image_proc stereo_image_proc
