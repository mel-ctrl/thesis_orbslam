currentPath=`pwd`
pathGT="/home/meltem/imow_line/visionTeam/Meltem/Datasets/Own/GT/van_adrichem/mc0003_20220619_071337_after_rain.txt"
pathResult="${currentPath}/evaluation/results/juno_stereo.pdf"
scriptPath="${currentPath}/evaluation/Scripts/evaluate_ate_scale.py"
trajPath="${currentPath}/CameraTrajectory.txt"
iniFAST=40
minFAST=14

python $scriptPath $pathGT $trajPath --plot $pathResult --iniFAST $iniFAST --minFAST $minFAST --verbose --offset -448504690000000

