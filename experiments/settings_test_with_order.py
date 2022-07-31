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
import matplotlib
matplotlib.use('TkAgg')
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
        self.histo_folder = self.stats_folder + "imgs/" + "histograms/"
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
        self.matches_img_nr = []
        self.sizes = [6, 12, 24, 48]
        self.fasttresh = [5, 10, 20, 40]
        self.n_features = 2000
        self.scale_factor = [1.1, 1.2, 1.3, 1.4]
        self.n_levels = [4, 8, 12, 16]

        self.n_features_effect = []
        self.scale_factor_effect = []
        self.n_levels_effect = []
        self.patch_size_effect = []
        self.fasttresh_effect = []
        self.inliers = []
                
        if self.dataset == "flourish" or self.dataset == "rosario" or self.dataset == "own":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-1:][0]).split('.')[0]
            if  self.dataset == "flourish":
                self.image_topic = "/sensor/camera/vi_sensor/left/image_raw"
                self.fps = 25
            elif self.dataset == "rosario":
                self.image_topic = "/stereo/left/image_raw"
                self.fps = 15
            elif self.dataset == "own":
                self.image_topic = "/daheng_camera_manager/left/image_rect"
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

        if not os.path.exists(self.save_stats_folder):
            os.makedirs(self.save_stats_folder)

        self.max_min_img_folder = self.stats_folder + "imgs/" + self.save_extension
        if not os.path.exists(self.max_min_img_folder):
            os.makedirs(self.max_min_img_folder)
        if not os.path.exists(self.histo_folder):
            os.makedirs(self.histo_folder)

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
                self.matches_img_nr.append(i)
            except:
                continue

    def filterMatches(self):  
        for image in range(len(self.matches)):
            matches_good = []
            if(self.matches[image]):
                #matchesMask = [[0,0] for i in range(len(self.matches[image]))]
                for i, match_pair in enumerate(self.matches[image]):
                    try:
                        m,n = match_pair
                        if m.distance < 0.75*n.distance:
                            #if 'matchesMask' in locals():
                            #    matchesMask[i]=[1,0]
                            matches_good.append(m)
                    except(ValueError):
                        pass
            homo_mat, matchesMask = self.findHomography(matches_good, image)
            self.inliers.append(matchesMask.count(1))

        if (matchesMask.count(1)) < 20:
            self.track_error += 1

    def findHomography(self, matches_good, img_pair):
        MIN_MATCH_COUNT = 10
        if len(matches_good)>MIN_MATCH_COUNT:
            kp2 = (self.keypoints[img_pair+1])
            kp1 = (self.keypoints[img_pair])

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches_good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        else:
            M=None
            n = len(matches_good)
            matchesMask = [0]*n
        return M, matchesMask

    def addImages(self, img):
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if self.equalize == True:
            img = self.equalizeHist(img)
        if self.ds_resolution == True:
            img = cv.resize(img, (self.target_width, self.target_height))
        self.images.append(img)

    def equalizeHist(self, img_gray):
        img = cv.equalizeHist(img_gray)
        return img

    def doMatching(self, orb):
        self.findKeyPoints(orb)
        self.matchKeyPointsBF() 
        self.filterMatches()

    def calcStats(self):
        stats_plot = plt.figure(1)
        y = self.inliers
        #std = round(np.std(y),1)
        mean = round(np.mean(y),1)
        #median = np.median(y)
        #spread = np.max(y)-np.min(y)
        min = np.min(y)
        min_img = self.images[self.matches_img_nr[np.argmin(y)]]
        max_img = self.images[self.matches_img_nr[np.argmax(y)]]


        return mean, min, min_img, max_img

    def clearResults(self):
        self.track_error = 0
        self.keypoints = []
        self.descriptor = []
        self.matches = []
        self.matches_good = []
        self.matches_img_nr = []
        self.inliers = []


    def plotHistogram(self, img, extension):
        plt.figure(0)
        histg = cv.calcHist([img],[0],None,[256],[0,256])
        plt.plot(histg)
        plt.savefig(self.histo_folder + self.save_extension.replace("/", "_") + extension)

    def plotEffectSettings(self):
        effect_plot= plt.figure(1)
        effect_plot, axes = plt.subplots(nrows = 5, ncols = 2, figsize=(10,8))
        effect_plot.suptitle(self.save_extension, size=8, y=1)
        
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


    def writeStatsToFile(self):
        n_features_mean = []
        n_features_min = [] 

        scale_factor_mean = []
        scale_factor_min = []

        n_levels_mean = []
        n_levels_min = []

        sizes_mean = []
        sizes_min = []

        fasttresh_mean = [] 
        fasttresh_min = []

        [n_features_mean.append(float(i[0])) for i in self.n_features_effect]
        [n_features_min.append(int(i[1])) for i in self.n_features_effect]

        [scale_factor_mean.append(float(i[0])) for i in self.scale_factor_effect]
        [scale_factor_min.append(int(i[1])) for i in self.scale_factor_effect]

        [n_levels_mean.append(float(i[0])) for i in self.n_levels_effect]
        [n_levels_min.append(int(i[1])) for i in self.n_levels_effect]

        [sizes_mean.append(float(i[0])) for i in self.patch_size_effect]
        [sizes_min.append(int(i[1])) for i in self.patch_size_effect]
        
        [fasttresh_mean.append(float(i[0])) for i in self.fasttresh_effect]
        [fasttresh_min.append(int(i[1])) for i in self.fasttresh_effect]

        '''
        data_dict = {'n_features': self.n_features, 'n_features_mean': n_features_mean, 'n_features_min': n_features_min, 'scale_factor': self.scale_factor, 'scale_factor_mean': scale_factor_mean, 'scale_factor_min': scale_factor_min, \
            'n_levels': self.n_levels, 'n_levels_mean': n_levels_mean, 'n_levels_min': n_levels_min, \
                'sizes': self.sizes, 'sizes_mean': sizes_mean, 'sizes_min': sizes_min,
                'fasttresh': self.fasttresh, 'fasttresh_mean': fasttresh_mean, 'fasttresh_min': fasttresh_min}
        data = {self.save_extension : data_dict}
        with open("/home/meltem/thesis_orbslam/experiments/settings_result.yaml", "a") as file:
            yaml.dump(data, file, default_flow_style=False)
        '''
        
        
        with open(self.stats_folder + "settings_" + self.save_extension.split('/')[0] + ".csv", 'a') as statsfile:
            writer = csv.writer(statsfile)
            writer.writerow([self.save_extension])
            header = ['parameter', 'setting', 'mean', 'min']
            writer.writerow(header)
            writer.writerow([])
            for i in range(len(self.fasttresh)):
                row = ["FAST Threshold", self.fasttresh[i], fasttresh_mean[i], fasttresh_min[i]]
                writer.writerow(row)
            writer.writerow([])
            for i in range(len(self.n_levels)):
                row = ["Number of levels", self.n_levels[i], n_levels_mean[i], n_levels_min[i]]
                writer.writerow(row)
            writer.writerow([])
            for i in range(len(self.scale_factor)):
                row = ["Scale factor", self.scale_factor[i], scale_factor_mean[i], scale_factor_min[i]]
                writer.writerow(row)
            writer.writerow([])
            for i in range(len(self.sizes)):
                row = ["Patch size", self.sizes[i], sizes_mean[i], sizes_min[i]]
                writer.writerow(row)
            writer.writerow([])
            writer.writerow([])
        statsfile.close()


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
        print("Read all images")


        for i in range(len(self.fasttresh)):
            orb = cv.ORB_create(fastThreshold=self.fasttresh[i], nfeatures=self.n_features)
            self.doMatching(orb)
            mean, min, min_img, max_img = self.calcStats()
            self.fasttresh_effect.append([mean,min])
            self.clearResults()
        
        besttresh_idx = np.argmax(np.asarray(self.fasttresh_effect)[:,1])

        for i in range(len(self.sizes)):
            orb = cv.ORB_create(edgeThreshold=self.sizes[i], patchSize=self.sizes[i], fastThreshold=self.fasttresh[besttresh_idx])
            self.doMatching(orb)
            mean, min, min_img, max_img = self.calcStats()
            self.patch_size_effect.append([mean,min])
            self.clearResults()
        
        bestsize_idx = np.argmax(np.asarray(self.patch_size_effect)[:,1])

        for i in range(len(self.scale_factor)):
            orb = cv.ORB_create(scaleFactor=self.scale_factor[i], edgeThreshold=self.sizes[bestsize_idx], patchSize=self.sizes[bestsize_idx], fastThreshold=self.fasttresh[besttresh_idx])
            self.doMatching(orb)
            mean, min, min_img, max_img = self.calcStats()
            self.scale_factor_effect.append([mean,min])
            self.clearResults()

        bestscale_idx = np.argmax(np.asarray(self.scale_factor_effect)[:,1])
        
        for i in range(len(self.n_levels)):
            orb = cv.ORB_create(nlevels=self.n_levels[i], scaleFactor=self.scale_factor[bestscale_idx], edgeThreshold=self.sizes[bestsize_idx], patchSize=self.sizes[bestsize_idx], fastThreshold=self.fasttresh[besttresh_idx])
            self.doMatching(orb)
            mean, min, min_img, max_img = self.calcStats()
            self.n_levels_effect.append([mean,min])
            self.clearResults()

        bestlevel_idx = np.argmax(np.asarray(self.n_levels_effect)[:,1])

        orb = cv.ORB_create(nlevels=self.n_levels[bestlevel_idx], scaleFactor=self.scale_factor[bestscale_idx], edgeThreshold=self.sizes[bestsize_idx], patchSize=self.sizes[bestsize_idx], fastThreshold=self.fasttresh[besttresh_idx])
        self.doMatching(orb)
        mean, min, min_img, max_img= self.calcStats()

        cv.imwrite(self.max_min_img_folder + "_min_img.jpg" , min_img)
        cv.imwrite(self.max_min_img_folder + "_max_img.jpg" , max_img)
        self.plotHistogram(min_img, "_min_img.png")
        self.plotHistogram(max_img, "_max_img.png")
        #settings_plot = self.plotEffectSettings()
        #settings_plot.savefig(self.stats_folder + self.save_extension + "_settings.png")
        self.writeStatsToFile()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''This script reads the dataset images or rosbag files and outputs the matches over time''')
    parser.add_argument('dataset', help='euroc, flourish, rosario, kitti, 4seasons, own')
    parser.add_argument('--source', help='specify source bag file or folder', default="")
    parser.add_argument('--equalize', help='Histogram equalization, True or False', default=False)
    parser.add_argument('--ds_fps', help='Downsample fps to equalize evaluation between datasets, True or False', default=False)
    parser.add_argument('--ds_resolution', help='Downsample resolution to equalize evaluation between datasets, True or False', default=False)
    parser.add_argument('--save_video', help='Save video with statistics', default=False)
    args = parser.parse_args()
    #args = parser.parse_args(["rosario", "--source", "/home/meltem/imow_line/visionTeam/Meltem/Datasets/Rosario/sequence01.bag", "--ds_fps", "False", "--ds_resolution", "False", "--save_video", "False"])
    object = ORB(args.dataset, args.source, args.equalize, args.ds_fps, args.ds_resolution, args.save_video)
    print("Settings set to equalize: {equalize}, downsample_fps: {ds_fps}, downsample_image: {ds_img}, save_video: {savevid}".format(equalize = args.equalize, ds_fps=args.ds_fps, ds_img=args.ds_resolution, savevid=args.save_video))
    object.main()
