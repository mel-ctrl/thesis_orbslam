currentPath=`pwd`
pathGT="${currentPath}/evaluation/Ground_truth/Rosario/sequence01_new_gt_left_cam_frame.txt"
pathResult="${currentPath}/evaluation/results/Rosario1_stereo.pdf"
scriptPath="${currentPath}/evaluation/Scripts/evaluate_ate_scale.py"
trajPath="${currentPath}/CameraTrajectory.txt"
iniFAST=40
minFAST=14

python $scriptPath $pathGT $trajPath --plot $pathResult --iniFAST $iniFAST --minFAST $minFAST --verbose

