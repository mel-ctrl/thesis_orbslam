#!/usr/bin/bash
evo_traj tum "/home/meltem/thesis_orbslam/KeyFrameTrajectoryTUM.txt" "/home/meltem/thesis_orbslam/experiments/stereo_VO/tum_trajectory.txt" --ref="kitti_converted_to_tum.txt" -p --plot_mode=xz --align --sync --correct_scale
