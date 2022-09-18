#!/usr/bin/bash
evo_traj tum "/home/meltem/thesis_orbslam/KeyFrameTrajectoryTUM.txt" --ref="kitti_converted_to_tum.txt" -p --plot_mode=xz --align --sync --correct_scale
