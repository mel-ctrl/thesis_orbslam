import csv
import matplotlib.pyplot as plt
import numpy as np

file = "/home/meltem/thesis_orbslam/den_boer_sunny_r09_thlow80_l6/matchingSTATS.txt"
save_name = file.rsplit('/',1)[0] + "/" + ((file.rsplit('/',1)[-1]).split('.')[0]).split('_',1)[-1] + ".png"
nlevels=list(file.rsplit('/',2)[-2].split('_')[-1])[-1]
th = ('').join(list(file.rsplit('/',2)[-2].split('_')[-2])[-2:])
nnratio = list(file.rsplit('/',2)[-2].split('_')[-3])[-2] + '.' + list(file.rsplit('/',2)[-2].split('_')[-3])[-1]
matches = []
inliers = []
features = []
x = []
with open(file, 'r') as stats:
    for line in stats:
        seq, ts, map, nrfeat, nrmatch, nrinl = line.split(",")
        features.append(int(nrfeat))
        matches.append(int(nrmatch))
        inliers.append(int(nrinl))
        x.append(int(seq))



fig, plot = plt.subplots(3, figsize = (6,6))
fig.suptitle("node level = {level}, TH_LOW = {th}, NNratio = {ratio}".format(level=nlevels, th=th, ratio=nnratio))
matches_under_min = [value for x, value in enumerate(matches) if value < 15]
matches_under_min_x = [x for x, value in enumerate(matches) if value < 15]

inliers_under_min = [value for x, value in enumerate(inliers) if value < 15]
inliers_under_min_x = [x for x, value in enumerate(inliers) if value < 15]

plot[0].scatter(x, features, s=1)
plot[0].scatter(np.argmin(features), np.min(features), s=5, color='red', label="min: " + str(np.min(features)))
plot[0].axhline(y=np.mean(features), color='g', linestyle='-', label="mean: " + str(round(np.mean(features),1)))
plot[0].set_xlabel("frames")
plot[0].set_ylabel('Number of features')

plot[1].scatter(x, matches, s=1)
plot[1].scatter(matches_under_min_x, matches_under_min, s=5, color='red', label="<15: " + str(sum(i < 15 for i in matches)))
plot[1].axhline(y=np.mean(matches), color='g', linestyle='-', label="mean: " + str(round(np.mean(matches),1)))
plot[1].set_xlabel("frames")
plot[1].set_ylabel('Number of matches')

plot[2].scatter(x, inliers, s=1)
plot[2].scatter(inliers_under_min_x, inliers_under_min, s=5, color='red', label="<10: " + str(sum(i < 10 for i in inliers)))
plot[2].set_ylabel('Number of inliers')
plot[2].axhline(y=np.mean(inliers), color='g', linestyle='-',label="mean: " + str(round(np.mean(inliers),1)))
plot[2].set_xlabel("frames")

plot[0].legend()
plot[1].legend()
plot[2].legend()
plt.savefig(save_name, format="png")