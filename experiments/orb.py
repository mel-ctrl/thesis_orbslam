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

class ORB:
    def __init__(self, dataset, source, equalize):
        self.source = source
        self.images = []
        self.keypoints = []
        self.descriptor = []
        self.matches = []
        self.matches_good = []
        self.dataset = dataset
        self.equalize = equalize
        self.stats_folder = "/home/meltem/thesis_orbslam/experiments/stats/"
        self.histo_folder = "/home/meltem/thesis_orbslam/experiments/histograms/"

        if self.dataset == "flourish" or self.dataset == "rosario" or self.dataset == "own":
            self.save_extention = self.dataset + "/" + (self.source.split('/')[-1:][0]).split('.')[0]
            if  self.dataset == "flourish":
                self.image_topic = "/sensor/camera/vi_sensor/left/image_raw"
            elif self.dataset == "rosario":
                self.image_topic = "/stereo/left/image_raw"
            elif self.dataset == "own":
                self.image_topic = "/daheng_camera_manager/left/image_rect"
        elif self.dataset == "euroc":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-5:][0])
            self.save_stats_folder = self.stats_folder + self.dataset
            self.save_hist_folder = self.histo_folder + self.dataset
        elif self.dataset == "seasons":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-4:][0]) + "/" + (self.source.split('/')[-3:][0])
            self.save_stats_folder = self.stats_folder + '/'.join(self.save_extension.split('/')[0:-2])
            self.save_hist_folder = self.histo_folder + '/'.join(self.save_extension.split('/')[0:-2])
        elif self.dataset == "kitti":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-2:][0])
            self.save_stats_folder = self.stats_folder + self.dataset
            self.save_hist_folder = self.histo_folder + self.dataset


        if not os.path.exists(self.save_stats_folder):
            os.makedirs(self.save_stats_folder)
        if not os.path.exists(self.save_hist_folder):
            os.makedirs(self.save_hist_folder)

    def findKeyPoints(self, orb):
        for img in self.images:
            kp, des = orb.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptor.append(des)

    def drawKeyPoints(self):
        plt.figure(3)
        img = cv.drawKeypoints(self.images[0], self.keypoints[0], None, color=(0,255,0), flags=0)
        plt.imshow(img)
        plt.show()

    def matchKeyPointsBF(self):
        bf = cv.BFMatcher()
        for i in range(len(self.images)-1):
            matches = bf.knnMatch(self.descriptor[i], self.descriptor[i+1], k=2)
            self.matches.append(matches)

    def matchKeyPointsFLANN(self):
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        for i in range(len(self.images)-1):
            matches = flann.knnMatch(self.descriptor[i], self.descriptor[i+1], k=2)
            for idx, match in enumerate(matches):
                if len(match) < 2:
                    matches.pop(idx)
            self.matches.append(matches)


    def drawMatches(self, plot):
        for image in range(len(self.images)-1):
            matches_good = []
            matchesMask = [[0,0] for i in range(len(self.matches[image]))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(self.matches[image]):
                if m.distance < 0.99*n.distance:
                    matchesMask[i]=[1,0]
                    matches_good.append(m)
            self.matches_good.append(matches_good)

            if plot==True:
                plt.figure(2)          
                draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
                img = cv.drawMatchesKnn(self.images[image],self.keypoints[image],self.images[image+1],self.keypoints[image+1],self.matches[image],None,**draw_params)
                plt.imshow(img)
                plt.show()


    def plotStats(self, patchsize, fasttresh):
        plt.figure(1)
        y = []
        xint = range(0, len(self.matches_good))
        for i in xint:
            y.append(len(self.matches_good[i]))
        plt.plot(xint, y)
        plt.axhline(np.mean(y), linestyle='--')
        print(pathlib.Path(__file__).parent.resolve())
        plt.title("Amount of good matches per frame, std: {std}, \n mean: {mean}, median: {median}, \n minimum: {min}, spread: {spread}".format(std=np.std(y), mean=np.mean(y), median=np.median(y), min=min(y), spread = max(y)-min(y)))
        plt.savefig(self.stats_folder + self.save_extension + "_patchsize_{patchsize}_fasttresh_{fasttresh}.png".format(patchsize=patchsize, fasttresh=fasttresh))
        plt.clf()
    def equalizeHist(self, img_gray):
        img = cv.equalizeHist(img_gray)
        return img

    def plotHistogram(self, img_gray):
        plt.figure(0)
        histg = cv.calcHist([img_gray],[0],None,[256],[0,256])
        plt.plot(histg)
        plt.savefig(self.histo_folder + self.save_extension + ".png")

    def main(self):
        if self.dataset == "flourish" or self.dataset == "rosario" or self.dataset == "own":
            bag = rosbag.Bag(self.bag_file, "r")
            bridge = CvBridge()
            i = 0
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if cv_img is not None:
                    if cv_img.shape[-1] == 3:
                        cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
                    if self.equalize == True:
                        cv_img = self.equalizeHist(cv_img)
                    self.images.append(cv_img)
                    #self.plotHistogram(cv_img)
                #if len(self.images) == 5:
                #    break
            bag.close()

        else:
            for filename in os.listdir(self.source):
                img = cv.imread(os.path.join(self.source,filename))
                if img is not None:
                    if img.shape[-1] == 3:
                        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    if self.equalize == True:
                        img = self.equalizeHist(img)
                    self.images.append(img)
                    #self.plotHistogram(img)

        sizes = [6, 12, 24, 48]
        fasttresh = [5, 10, 20, 50]
        for i in range(len(sizes)):
            for j in range(len(fasttresh)):
                orb = cv.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=sizes[i], firstLevel=0, WTA_K=2, scoreType=ORB_HARRIS_SCORE , patchSize=sizes[i], fastThreshold=fasttresh[j])
                self.findKeyPoints(orb)
                #self.drawKeyPoints()
                self.matchKeyPointsBF()
                #self.matchKeyPointsFLANN()
                
                self.drawMatches(plot=False)
                self.plotStats(sizes[i], fasttresh[j])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''This script reads the dataset images or rosbag files and outputs the matches over time''')
    parser.add_argument('--dataset', help='euroc, flourish, rosario, kitti, 4seasons, own')
    parser.add_argument('--source', help='specify source bag file or folder', default="")
    parser.add_argument('--equalize', help='Histogram equalization, True or False', default=False)
    #args = parser.parse_args()
    args = parser.parse_args(["--dataset", "euroc", "--source", "/media/meltem/moo/Meltem/Thesis/Datasets/EuRoC/MH01/mav0/cam0/data/"])
    object = ORB(args.dataset, args.source, args.equalize)
    object.main()