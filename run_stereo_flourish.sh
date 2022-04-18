#!/bin/bash

rosrun ORB_SLAM3 Stereo ./Vocabulary/ORBvoc.txt ./Examples/Stereo/Flourish.yaml false /camera/left/image_raw:=sensor/camera/vi_sensor/left/image_rect /camera/right/image_raw:=sensor/camera/vi_sensor/right/image_rect
