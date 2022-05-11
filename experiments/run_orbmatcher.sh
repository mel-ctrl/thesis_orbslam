#!/bin/bash

currentPath=`pwd`
scriptPath="${currentPath}/orb.py"
datasetPath= "/media/meltem/moo/Meltem/Thesis/Datasets"
rosarioPath="${datasetPath}/Rosario"
flourishPath="${datasetPath}/Flourish"
eurocPath="${datasetPath}/EuRoC"
seasonsPath="${datasetPath}/4seasons"
kittiPath="${datasetPath}/kitti"
ownPath="${datasetPath}/own"

python $scriptPath rosario --bag_file "${rosarioPath}/sequence01.bag"
python $scriptPath rosario --bag_file "${rosarioPath}/sequence02.bag"
python $scriptPath rosario --bag_file "${rosarioPath}/sequence03.bag"
python $scriptPath rosario --bag_file "${rosarioPath}/sequence04.bag"
python $scriptPath rosario --bag_file "${rosarioPath}/sequence05.bag"
python $scriptPath rosario --bag_file "${rosarioPath}/sequence06.bag"

python $scriptPath flourish --bag_file "${flourishPath}/DatasetA.bag"
python $scriptPath flourish --bag_file "${flourishPath}/DatasetB.bag"

python $scriptPath own --bag_file "${ownPath}/concrete.bag"
python $scriptPath own --bag_file "${ownPath}/farm.bag"
python $scriptPath own --bag_file "${ownPath}/grass.bag"

python $scriptPath euroc --source_dir "${eurocPath}/MH01/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/MH02/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/MH03/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/MH04/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/MH05/mav0/cam0/data"

python $scriptPath euroc --source_dir "${eurocPath}/V101/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/V102/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/V103/mav0/cam0/data"

python $scriptPath euroc --source_dir "${eurocPath}/V201/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/V202/mav0/cam0/data"
python $scriptPath euroc --source_dir "${eurocPath}/V203/mav0/cam0/data"

python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/00/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/01/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/02/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/03/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/04/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/05/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/06/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/07/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/08/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/09/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/10/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/11/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/12/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/13/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/14/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/15/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/16/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/17/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/18/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/19/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/20/image_2"
python $scriptPath kitti --source_dir "${kittiPath}/data_odometry_color/dataset/sequences/21/image_2"

python $scriptPath seasons --source_dir "${seasonsPath}/highway/loop1/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/highway/loop2/distorted_images/cam0"

python $scriptPath seasons --source_dir "${seasonsPath}/office_loop/loop1/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/office_loop/loop2/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/office_loop/loop3/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/office_loop/loop4/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/office_loop/loop5/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/office_loop/loop6/distorted_images/cam0"

python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop1/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop2/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop3/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop4/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop5/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop6/distorted_images/cam0"
python $scriptPath seasons --source_dir "${seasonsPath}/neighborhood/loop7/distorted_images/cam0"



