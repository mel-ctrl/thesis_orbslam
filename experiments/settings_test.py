from ast import parse
from cv2 import ORB_HARRIS_SCORE, imshow
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import yaml
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import pathlib
import csv

class ORB:
    def __init__(self, dataset, source, equalize, ds_fps, ds_resolution, save_video):
        self.source = source
        self.images = []
        self.keypoints = []
        self.descriptor = []
        self.matches = []
        self.matches_good = []
        self.dataset = dataset
        self.equalize = bool(equalize)
        self.stats_folder = "/home/meltem/thesis_orbslam/experiments/stats/"
        self.histo_folder = "/home/meltem/thesis_orbslam/experiments/histograms/"
        self.statsfile =  "/home/meltem/thesis_orbslam/experiments/stats.yaml"
        self.save_stats_folder = self.stats_folder + self.dataset
        self.save_hist_folder = self.histo_folder + self.dataset
        self.target_fps = 10
        self.target_width, self.target_height = 672, 376
        self.ds_fps = bool(ds_fps)
        self.ds_resolution = bool(ds_resolution)
        self.track_error = 0
        self.save_video = bool(save_video)
        self.effectSettings = []

        self.sizes = [6, 12, 24, 48]
        self.fasttresh = [5, 10, 20, 40]
        self.n_features = [500, 1000, 2000, 3000]
        self.scale_factor = [1.1, 1.2, 1.3, 1.4]
        self.n_levels = [4, 8, 12, 16]

        self.n_features_effect = []
        self.scale_factor_effect = []
        self.n_levels_effect = []
        self.patch_size_effect = []
        self.fasttresh_effect = []
                
        if self.dataset == "flourish" or self.dataset == "rosario" or self.dataset == "own":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-1:][0]).split('.')[0]
            if  self.dataset == "flourish":
                self.image_topic = "/sensor/camera/vi_sensor/left/image_raw"
                self.fps = 25
            elif self.dataset == "rosario":
                self.image_topic = "/stereo/left/image_raw"
                self.fps = 15
            elif self.dataset == "own":
                self.image_topic = "/daheng_camera_manager/left/image_raw"
                self.fps = 10
        elif self.dataset == "euroc":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-4:][0])
            self.fps = 20
        elif self.dataset == "seasons":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-4:][0]) + "/" + (self.source.split('/')[-3:][0])
            self.save_stats_folder = self.stats_folder + '/'.join(self.save_extension.split('/')[0:-1])
            self.save_hist_folder = self.histo_folder + '/'.join(self.save_extension.split('/')[0:-1])
            self.fps = 30
        elif self.dataset == "kitti":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-2:][0])
            self.fps = 10


    def findKeyPoints(self, orb):
        for img in self.images:
            kp, des = orb.detectAndCompute(img, None)
            if len(kp) != 0 and len(des) != 0:
                self.keypoints.append(kp)
                self.descriptor.append(des)

    def drawKeyPoints(self):
        keypoint_plot = plt.figure(3)
        img = cv.drawKeypoints(self.images[0], self.keypoints[0], None, color=(0,255,0), flags=0)
        plt.imshow(img)
        keypoint_plot.show()

    def matchKeyPointsBF(self):
        bf = cv.BFMatcher()
        for i in range(len(self.images)-1):
            try:
                matches = bf.knnMatch(self.descriptor[i], self.descriptor[i+1], k=2)
                self.matches.append(matches)
            except:
                continue

    def filterMatches(self):
        for image in range(len(self.images)-1):
            matches_good = []
            matchesMask = [[0,0] for i in range(len(self.matches[image]))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(self.matches[image]):
                if m.distance < 0.75*n.distance:
                    matchesMask[i]=[1,0]
                    matches_good.append(m)
            self.matches_good.append(matches_good)

        if len(self.matches_good) < 20:
            self.track_error += 1
        
    def addImages(self, img):
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if self.equalize == True:
            img = self.equalizeHist(img)
        if self.ds_resolution == True:
            img = cv.resize(img, (self.target_width, self.target_height))
        self.images.append(img)

    def doMatching(self, orb):
        self.findKeyPoints(orb)
        self.matchKeyPointsBF() 
        self.filterMatches()

    def calcStats(self):
        stats_plot = plt.figure(1)
        y = []
        xint = range(0, len(self.matches_good))
        for i in xint:
            y.append(len(self.matches_good[i]))

        std = round(np.std(y),1)
        mean = round(np.mean(y),1)
        median = np.median(y)
        spread = np.max(y)-np.min(y)
        min = np.min(y)

        return mean, min

    def clearResults(self):
        self.track_error = 0
        self.keypoints = []
        self.descriptor = []
        self.matches = []
        self.matches_good = []



    def plotEffectSettings(self):
        effect_plot= plt.figure(1)
        effect_plot, axes = plt.subplots(nrows = 5, ncols = 2, figsize=(8,6))
        effect_plot.suptitle(self.save_extension)
        
        for i in range(len(self.n_features)):
            axes[0][0].scatter(self.n_features[i], self.n_features_effect[i][0])
            axes[0][1].scatter(self.n_features[i], self.n_features_effect[i][1])
            axes[1][0].scatter(self.scale_factor[i], self.scale_factor_effect[i][0])
            axes[1][1].scatter(self.scale_factor[i], self.scale_factor_effect[i][1])
            axes[2][0].scatter(self.n_levels[i], self.n_levels_effect[i][0])
            axes[2][1].scatter(self.n_levels[i], self.n_levels_effect[i][1])
            axes[3][0].scatter(self.sizes[i], self.patch_size_effect[i][0])
            axes[3][1].scatter(self.sizes[i], self.patch_size_effect[i][1])
            axes[4][0].scatter(self.fasttresh[i], self.fasttresh_effect[i][0])
            axes[4][1].scatter(self.fasttresh[i], self.fasttresh_effect[i][1])

        
        axes[0][0].set_xlabel('n_features parameter')
        axes[0][0].set_ylabel('Mean')
        axes[0][0].set_xticks(self.n_features)

        axes[0][1].set_xlabel('n_features parameter')
        axes[0][1].set_ylabel('Minimum')
        axes[0][1].set_xticks(self.n_features)


        axes[1][0].set_xlabel('scale_factor parameter')
        axes[1][0].set_ylabel('Mean')
        axes[1][0].set_xticklabels(self.scale_factor)
        axes[1][0].set_xticks(self.scale_factor)

        axes[1][1].set_xlabel('scale factor parameter')
        axes[1][1].set_ylabel('Minimum')
        axes[1][1].set_xticks(self.scale_factor)


        axes[2][0].set_xlabel('n_levels parameter')
        axes[2][0].set_ylabel('Mean')
        axes[2][0].set_xticks(self.n_levels)

        axes[2][1].set_xlabel('n_levels parameter')
        axes[2][1].set_ylabel('Minimum')
        axes[2][1].set_xticks(self.n_levels)

        
        axes[3][0].set_xlabel('patch size parameter')
        axes[3][0].set_ylabel('Mean')
        axes[3][0].set_xticks(self.sizes)

        axes[3][1].set_xlabel('patch size parameter')
        axes[3][1].set_ylabel('Minimum')
        axes[3][1].set_xticks(self.sizes)


        axes[4][0].set_xlabel('FAST treshold parameter')
        axes[4][0].set_ylabel('Mean')
        axes[4][0].set_xticks(self.fasttresh)

        axes[4][1].set_xlabel('FAST treshold parameter')
        axes[4][1].set_ylabel('Minimum')
        axes[4][1].set_xticks(self.fasttresh)
        
        plt.tight_layout()

        return effect_plot

    def main(self):
        print("Dataset: {dataset}".format(dataset=self.save_extension))
        i = 0
        if self.dataset == "flourish" or self.dataset == "rosario" or self.dataset == "own":
            bag = rosbag.Bag(self.source, "r")
            bridge = CvBridge()
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img is not None and len(img) != 0:
                    if self.ds_fps and (self.fps != self.target_fps):
                        i +=1
                        if i % (self.fps/(self.fps-self.target_fps)) != 0:
                            self.addImages(img)
                    else:
                        self.addImages(img)
            bag.close()

        else:
            for filename in sorted(os.listdir(self.source)):
                img = cv.imread(os.path.join(self.source,filename))
                if img is not None and len(img) != 0:
                    if self.ds_fps and (self.fps != self.target_fps):
                        i +=1
                        if i % (self.fps/(self.fps-self.target_fps)) != 0:
                            self.addImages(img)
                    else:
                        self.addImages(img)
                if len(self.images) == 3:
                    break

        print("Read all images")


        for i in range(len(self.n_features)):
            orb = cv.ORB_create(nfeatures=self.n_features[i])
            self.doMatching(orb)
            mean, min = self.calcStats()
            self.n_features_effect.append([mean,min])
            self.clearResults()

            
        for i in range(len(self.scale_factor)):
            orb = cv.ORB_create(scaleFactor=self.scale_factor[i])
            self.doMatching(orb)
            mean, min = self.calcStats()
            self.scale_factor_effect.append([mean,min])
            self.clearResults()

        for i in range(len(self.n_levels)):
            orb = cv.ORB_create(nlevels=self.n_levels[i])
            self.doMatching(orb)
            mean, min = self.calcStats()
            self.n_levels_effect.append([mean,min])
            self.clearResults()

        for i in range(len(self.sizes)):
            orb = cv.ORB_create(edgeThreshold=self.sizes[i], patchSize=self.sizes[i])
            self.doMatching(orb)
            mean, min = self.calcStats()
            self.patch_size_effect.append([mean,min])
            self.clearResults()

        for i in range(len(self.fasttresh)):
            orb = cv.ORB_create(fastThreshold=self.fasttresh[i])
            self.doMatching(orb)
            mean, min = self.calcStats()
            self.fasttresh_effect.append([mean,min])
            self.clearResults()

        settings_plot = self.plotEffectSettings()
        settings_plot.savefig(self.stats_folder + self.save_extension + "_settings.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''This script reads the dataset images or rosbag files and outputs the matches over time''')
    parser.add_argument('dataset', help='euroc, flourish, rosario, kitti, 4seasons, own')
    parser.add_argument('--source', help='specify source bag file or folder', default="")
    parser.add_argument('--equalize', help='Histogram equalization, True or False', default=False)
    parser.add_argument('--ds_fps', help='Downsample fps to equalize evaluation between datasets, True or False', default=False)
    parser.add_argument('--ds_resolution', help='Downsample resolution to equalize evaluation between datasets, True or False', default=False)
    parser.add_argument('--save_video', help='Save video with statistics', default=False)
    args = parser.parse_args()
    #args = parser.parse_args(["kitti", "--source", "/home/meltem/imow_line/visionTeam/Meltem/Datasets/kitti/data_odometry_color/dataset/sequences/00/image_2", "--ds_fps", "False", "--ds_resolution", "False", "--save_video", "False"])
    object = ORB(args.dataset, args.source, args.equalize, args.ds_fps, args.ds_resolution, args.save_video)
    print("Settings set to equalize: {equalize}, downsample_fps: {ds_fps}, downsample_image: {ds_img}, save_video: {savevid}".format(equalize = args.equalize, ds_fps=args.ds_fps, ds_img=args.ds_resolution, savevid=args.save_video))
    object.main()