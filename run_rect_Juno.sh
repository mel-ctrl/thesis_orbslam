#!/bin/bash
source /opt/ros/noetic/setup.bash
cd /home/meltem/catkin_ws
source devel/setup.bash
ROS_NAMESPACE=/daheng_camera_manager rosrun stereo_image_proc stereo_image_proc approximate_sync:=true
