#!/usr/bin/bash
#seq="07"
#evo_traj tum "/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/ORB-SLAM3.txt" "/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/matching/VO_with_feature_matcher.txt" "/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/"${seq}"/optical flow/VO_with_optical_flow.txt" --ref="/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/GT.txt" -p --plot_mode=xz --align --sync --correct_scale --align_origin

#evo_ape tum /home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/GT.txt /home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/ORB-SLAM3.txt -va #--plot --plot_mode xz

#evo_ape tum /home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/GT.txt /home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/matching/VO_with_feature_matcher.txt -va #--plot --plot_mode xz

#evo_ape tum /home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/GT.txt "/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/optical flow/VO_with_optical_flow.txt" -va #--plot --plot_mode xz

#evo_traj tum "/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/00/noisy/smoothed_disparity_map.txt" "/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/00/noisy/noisy_disparity_map.txt" --ref="/home/meltem/thesis_orbslam/experiments/results/kitti/KITTI/${seq}/GT.txt" -p --plot_mode=xz --align --sync --correct_scale  --align_origin

#evo_ape tum /home/meltem/thesis_orbslam/experiments/results/KITTI/${seq}/GT.txt /home/meltem/thesis_orbslam/experiments/results/KITTI/00/noisy/smoothed_disparity_map.txt -va #--plot --plot_mode xz

#evo_ape tum /home/meltem/thesis_orbslam/experiments/results/KITTI/${seq}/GT.txt "/home/meltem/thesis_orbslam/experiments/results/KITTI/00/noisy/noisy_disparity_map.txt" -va #--plot --plot_mode xz


dataset="den_boer_after_rain"
evo_traj tum "/media/meltem/T7/${dataset}/VO_with_feature_matcher.txt" "/media/meltem/T7/"${dataset}"/VO_with_optical_flow.txt" --ref="/media/meltem/T7/"${dataset}"/GT.txt" -p --plot_mode=xy --correct_scale --align --sync --n_to_align 200

evo_ape tum /media/meltem/T7/${dataset}/GT.txt "/media/meltem/T7/"${dataset}"/VO_with_feature_matcher.txt"

evo_ape tum /media/meltem/T7/${dataset}/GT.txt "/media/meltem/T7/"${dataset}"/VO_with_optical_flow.txt"
