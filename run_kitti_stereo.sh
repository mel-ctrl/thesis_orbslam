#!/bin/bash

pathDatasetKITTI='/media/meltem/moo/kitti/data_odometry_color'
echo "Launching sequence2 with Stereo sensor"
./Examples/Stereo/stereo_kitti ./Vocabulary/ORBvoc.txt ./Examples_old/Stereo/KITTI00-02.yaml "$pathDatasetKITTI"/dataset/sequences/02

