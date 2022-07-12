currentPath=`pwd`
pathGT="/home/meltem/mc0038_20220613_095019_sunny.txt"
pathResult="${currentPath}/evaluation/results/juno_stereo.pdf"
scriptPath="${currentPath}/evaluation/Scripts/evaluate_ate_scale.py"
trajPath="${currentPath}/CameraTrajectory.txt"
iniFAST=40
minFAST=14

python $scriptPath $pathGT $trajPath --plot $pathResult --iniFAST $iniFAST --minFAST $minFAST --verbose --offset -957502620000000

