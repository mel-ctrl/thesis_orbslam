currentPath=`pwd`
pathGT="/media/meltem/moo/kitti/GT/02.txt"
pathResult="${currentPath}/evaluation/results/kitti_stereo.pdf"
scriptPath="${currentPath}/evaluation/Scripts/evaluate_ate_scale.py"
trajPath="${currentPath}/CameraTrajectory.txt"
iniFAST=40
minFAST=14

python $scriptPath $pathGT $trajPath --plot $pathResult --iniFAST $iniFAST --minFAST $minFAST --verbose #--offset -957502620000000

