import csv
import matplotlib.pyplot as plt
import numpy as np

file = "/home/meltem/thesis_orbslam/experiments/bowlevels_effect/matchingSTATS_adrichem_sunny_2.txt"
save_name = file.rsplit('/',1)[0] + "/" + ((file.rsplit('/',1)[-1]).split('.')[0]).split('_',1)[-1] + ".png"
nlevels=file.rsplit('_',1)[-1].split('.')[0]

matches = []
inliers = []
features = []
with open(file, 'r') as stats:
    for line in stats:
        seq, ts, map, nrfeat, nrmatch, nrinl = line.split(",")
        features.append(int(nrfeat))
        matches.append(int(nrmatch))
        inliers.append(int(nrinl))



fig, plot = plt.subplots(3, figsize = (6,6))
fig.suptitle("Feature matching statistics \n node level = {level}".format(level=nlevels))


x= np.linspace(1,len(matches),len(matches))

plot[0].scatter(x, features)
plot[0].axhline(y=np.mean(features), color='g', linestyle='-', label="mean: " + str(round(np.mean(features),1)))
plot[0].set_xlabel("frames")
plot[0].set_ylabel('Number of features')

plot[1].scatter(x, matches)
plot[1].axhline(y=np.mean(matches), color='g', linestyle='-', label="mean: " + str(round(np.mean(matches),1)))
plot[1].set_xlabel("frames")
plot[1].set_ylabel('Number of matches')

plot[2].scatter(x, inliers)
plot[2].set_ylabel('Number of inliers')
plot[2].axhline(y=np.mean(inliers), color='g', linestyle='-',label="mean: " + str(round(np.mean(inliers),1)))
plot[2].set_xlabel("frames")

plot[0].legend()
plot[1].legend()
plot[2].legend()
plt.savefig(save_name, format="png")