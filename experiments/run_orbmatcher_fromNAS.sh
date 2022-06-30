#!/bin/bash

currentPath=`pwd`
scriptPath="${currentPath}/settings_test.py"
datasetPath="/home/meltem/imow_line/visionTeam/Meltem/Datasets"
rosarioPath="${datasetPath}/Rosario"
flourishPath="${datasetPath}/Flourish"
eurocPath="${datasetPath}/EuRoC"
seasonsPath="${datasetPath}/4seasons"
kittiPath="${datasetPath}/kitti"
ownPath="${datasetPath}/Own"
#otherArgs="--ds_fps True"
otherArgs=""


python $scriptPath rosario --source "${rosarioPath}/sequence01.bag" $otherArgs
python $scriptPath rosario --source "${rosarioPath}/sequence02.bag" $otherArgs
python $scriptPath rosario --source "${rosarioPath}/sequence03.bag" $otherArgs
python $scriptPath rosario --source "${rosarioPath}/sequence04.bag" $otherArgs
python $scriptPath rosario --source "${rosarioPath}/sequence05.bag" $otherArgs
python $scriptPath rosario --source "${rosarioPath}/sequence06.bag" $otherArgs

python $scriptPath flourish --source "${flourishPath}/DatasetA.bag" $otherArgs
python $scriptPath flourish --source "${flourishPath}/DatasetB.bag" $otherArgs

python $scriptPath own --source "${ownPath}/mc0006_20220523_123903_barn_rainy_corrected.bag" $otherArgs
python $scriptPath own --source "${ownPath}/mc0006_20220523_123903_grass_rainy_corrected.bag" $otherArgs
python $scriptPath own --source "${ownPath}/mc0006_20220523_123903_road_rainy_corrected.bag" $otherArgs

python $scriptPath own --source "${ownPath}/mc0006_20220523_232418_barn_dark_corrected.bag" $otherArgs
python $scriptPath own --source "${ownPath}/mc0006_20220523_232418_grass_dark_corrected.bag" $otherArgs
python $scriptPath own --source "${ownPath}/mc0006_20220523_232418_road_dark_corrected.bag" $otherArgs

python $scriptPath own --source "${ownPath}/mc0006_20220523_232418_grass_normal_corrected.bag" $otherArgs
python $scriptPath own --source "${ownPath}/mc0006_20220523_232418_road_normal_corrected.bag" $otherArgs

python $scriptPath own --source "${ownPath}/den_boer_mc0038_20220613_095019_sunny.bag" $otherArgs
python $scriptPath own --source "${ownPath}/den_boer_mc0038_20220615_005023_dark.bag" $otherArgs
python $scriptPath own --source "${ownPath}/den_boer_mc0038_20220619_075057_rain.bag" $otherArgs
python $scriptPath own --source "${ownPath}/den_boer_mc0038_20220619_080430_after_rain.bag" $otherArgs
python $scriptPath own --source "${ownPath}/van_adrichem_mc0003_20220619_071337_after_rain.bag" $otherArgs
python $scriptPath own --source "${ownPath}/van_adrichem_mc0003_20220621_161250_sunny.bag" $otherArgs
python $scriptPath own --source "${ownPath}/van_adrichem_mc0003_20220622_000215_dark.bag" $otherArgs

python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/00/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/01/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/02/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/03/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/04/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/05/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/06/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/07/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/08/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/09/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/10/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/11/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/12/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/13/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/14/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/15/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/16/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/17/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/18/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/19/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/20/image_2" $otherArgs
python $scriptPath kitti --source "${kittiPath}/data_odometry_color/dataset/sequences/21/image_2" $otherArgs

python $scriptPath euroc --source "${eurocPath}/MH01/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/MH02/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/MH03/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/MH04/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/MH05/mav0/cam0/data" $otherArgs

python $scriptPath euroc --source "${eurocPath}/V101/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/V102/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/V103/mav0/cam0/data" $otherArgs

python $scriptPath euroc --source "${eurocPath}/V201/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/V202/mav0/cam0/data" $otherArgs
python $scriptPath euroc --source "${eurocPath}/V203/mav0/cam0/data" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/highway/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/highway/loop2/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/office_loop/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/office_loop/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/office_loop/loop3/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/office_loop/loop4/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/office_loop/loop5/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/office_loop/loop6/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop3/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop4/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop5/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop6/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/neighborhood/loop7/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/business_campus/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/business_campus/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/business_campus/loop3/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/city_loop/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/city_loop/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/city_loop/loop3/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/country_side/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/country_side/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/country_side/loop3/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/country_side/loop4/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/maximalaneum/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/maximalaneum/loop2/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/old_town/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/old_town/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/old_town/loop3/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/old_town/loop4/distorted_images/cam0" $otherArgs

python $scriptPath seasons --source "${seasonsPath}/parking_garage/loop1/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/parking_garage/loop2/distorted_images/cam0" $otherArgs
python $scriptPath seasons --source "${seasonsPath}/parking_garage/loop3/distorted_images/cam0" $otherArgs

