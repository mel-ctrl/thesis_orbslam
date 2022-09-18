#!/usr/bin/bash
#"/home/meltem/thesis_orbslam/KeyFrameTrajectoryTUM.txt"
python3 ~/thesis_orbslam/experiments/evo/contrib/kitti_poses_and_timestamps_to_trajectory.py "/media/meltem/moo/kitti/GT/02.txt" "/media/meltem/moo/kitti/data_odometry_color/dataset/sequences/02/times.txt" "kitti_converted_to_tum.txt"

