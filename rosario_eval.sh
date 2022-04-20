currentPath=`pwd`
pathGT="${currentPath}/evaluation/Ground_truth/Rosario/sequence03_new_gt_left_cam_frame.txt"
pathResult="${currentPath}/evaluation/results/Rosario3_stereo.pdf"
scriptPath="${currentPath}/evaluation/Scripts/evaluate_ate_scale.py"
trajPath="${currentPath}/CameraTrajectory.txt"
iniFAST=20
minFAST=7

python $scriptPath $pathGT $trajPath --plot $pathResult --iniFAST $iniFAST --minFAST $minFAST --verbose

