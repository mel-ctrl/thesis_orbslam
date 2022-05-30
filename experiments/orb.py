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
        self.blur = []
        self.flow = []
        self.target_fps = 10
        self.target_width, self.target_height = 672, 376
        self.ds_fps = bool(ds_fps)
        self.ds_resolution = bool(ds_resolution)
        self.track_error = 0
        self.save_video = bool(save_video)
        
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

        video_dir = self.stats_folder + "videos/" + self.save_extension
        if not os.path.exists(self.save_stats_folder):
            os.makedirs(self.save_stats_folder)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

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


    def filterMatches(self, plot, out):
        for image in range(len(self.images)-1):
            matches_good = []
            matchesMask = [[0,0] for i in range(len(self.matches[image]))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(self.matches[image]):
                if m.distance < 0.8*n.distance:
                    matchesMask[i]=[1,0]
                    matches_good.append(m)
            self.matches_good.append(matches_good)

            if plot or self.save_video:
                draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
                img = cv.drawMatchesKnn(self.images[image],self.keypoints[image],self.images[image+1],self.keypoints[image+1],self.matches[image],None,**draw_params)
                if plot==True:
                    plt.figure(2)          
                    plt.imshow(img)
                    plt.show()
                if self.save_video:
                    out.write(img)


        if len(self.matches_good) < 20:
            self.track_error += 1
        
        out.release()
        cv.destroyAllWindows()
        

    def calculateFlow(self):
        for i in range(len(self.images)-1):
            flow = cv.calcOpticalFlowFarneback(self.images[i], self.images[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            mag, ang = self.cartToPol(flow[..., 0], flow[..., 1])
            self.flow.append(np.mean(mag))


    def cartToPol(self, x, y):
        ang = np.arctan2(y, x)
        mag = np.hypot(x, y)
        return mag, ang


    def variance_of_laplacian(self, image):
        return cv.Laplacian(image, cv.CV_64F).var()

    def writeStatsToFile(self, match_std, match_mean, match_median, match_minimum, match_spread, patchsize, fasttresh):
        blur_std = round(np.std(self.blur),1)
        blur_mean = round(np.mean(self.blur),1)
        blur_median=np.median(self.blur)
        blur_maximum=np.max(self.blur)
        blur_spread = np.max(self.blur)-np.min(self.blur)

        flow_std = round(np.std(self.flow),1)
        flow_mean = round(np.mean(self.flow),1)
        flow_median=np.median(self.flow)
        flow_maximum=np.max(self.flow)
        flow_spread = np.max(self.flow)-np.min(self.flow)
        
        data_dict = {'track_error': int(self.track_error), 'match_mean': float(match_mean), 'match_std': float(match_std), 'match_median': float(match_median), 'match_minimum': int(match_minimum), \
            'match_spread': int(match_spread), 'patchsize': int(patchsize), 'fasttresh': int(fasttresh), 'blur_mean': float(blur_mean), 'blur_std': float(blur_std), 'blur_median': float(blur_median), 'blur_maximum': float(blur_maximum), 'blur_spread': float(blur_spread), 'flow_mean': float(flow_mean), 'flow_std': float(flow_std), 'flow_median': float(flow_median), 'flow_maximum': float(flow_maximum), 'flow_spread': float(flow_spread)}
        #data = {self.save_extension : {'Matching':matching_dict, 'Blur':blur_dict, 'Flow': flow_dict}}
        data = {self.save_extension : data_dict}
        with open(self.statsfile, "a") as file:
            yaml.dump(data, file)

    def plotStats(self, patchsize, fasttresh):
        stats_plot = plt.figure(1)
        y = []
        xint = range(0, len(self.matches_good))
        for i in xint:
            y.append(len(self.matches_good[i]))
        plt.plot(xint, y)
        plt.axhline(np.mean(y), linestyle='--')
        
        print(pathlib.Path(__file__).parent.resolve())

        std = round(np.std(y),1)
        mean = round(np.mean(y),1)
        median = np.median(y)
        spread = np.max(y)-np.min(y)
        min = np.min(y)

        plt.title("std: {std}, mean: {mean}, median: {median}, \n minimum: {min}, spread: {spread}, \n patchsize: {patchsize}. fasttresh: {fasttresh}".format(std=std, mean=mean, median=median, min=min, spread = spread, patchsize=patchsize, fasttresh=fasttresh))
        
        return std, mean, median, spread, min, stats_plot
        
    def equalizeHist(self, img_gray):
        img = cv.equalizeHist(img_gray)
        return img

    def plotHistogram(self, img_gray):
        plt.figure(0)
        histg = cv.calcHist([img_gray],[0],None,[256],[0,256])
        plt.plot(histg)
        plt.savefig(self.histo_folder + self.save_extension + ".png")

    def addImages(self, img):
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if self.equalize == True:
            img = self.equalizeHist(img)
        if self.ds_resolution:
            img = cv.resize(img, (self.target_width, self.target_height))
        self.images.append(img)
        self.blur.append(self.variance_of_laplacian(img))

    def plotEffectSettings(self, settings, mean, index, minimum):
        effect_plot= plt.figure(4)
        effect_plot, axes = plt.subplots(nrows = 1, ncols = 2, sharex=True)
        
        axes[0].scatter(index, mean)
        axes[0].set_xlabel('Setting')
        axes[0].set_ylabel('Mean')
        axes[0].set_xticks(index)
        axes[0].set_xticklabels(settings,rotation=90)
        axes[1].scatter(index, minimum)
        axes[1].set_xlabel('Setting')
        axes[1].set_ylabel('Minimum')
        axes[1].set_xticks(index)
        axes[1].set_xticklabels(settings,rotation=90)

        plt.tight_layout()
        return effect_plot
    
    def removeNonBestVideos(self, patchsize, fasttresh):
        folder = self.stats_folder + "videos/" + self.save_extension
        for file in os.listdir(folder):
            if (str(patchsize) + "_" + str(fasttresh) + ".avi") in file:
                continue
            else:
                os.remove(folder + "/" + file)


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
            for filename in os.listdir(self.source):
                img = cv.imread(os.path.join(self.source,filename))
                if img is not None and len(img) != 0:
                    if self.ds_fps and (self.fps != self.target_fps):
                        i +=1
                        if i % (self.fps/(self.fps-self.target_fps)) != 0:
                            self.addImages(img)
                    else:
                        self.addImages(img)
                if len(self.images) == 50:
                    break

        print("Read all images")

        print('Calculating optical flow between all images')
        self.calculateFlow()

        sizes = [6, 12, 24, 48]
        fasttresh = [5, 10, 20, 40]
        best_std = 0
        best_mean = 0
        best_median = 0
        best_spread = 0
        best_min = 0
        best_patchsize = 0
        best_fasttresh = 0
        best_plt = None
        index = 0
        effect_plot_index = []
        effect_plot_settings = []
        effect_plot_match_mean = []
        effect_plot_match_min = []
        for i in range(len(sizes)):
            for j in range(len(fasttresh)):
                if self.save_video:
                    fourcc = cv.VideoWriter_fourcc(*'MJPG')
                    out = cv.VideoWriter((self.stats_folder + "videos/" + self.save_extension + "/" + str(sizes[i]) + "_" + str(fasttresh[j]) + ".avi"), fourcc, self.target_fps, (img.shape[1]*2, img.shape[0]))
                index+=1
                print("Matching features for patch size {size} and fast treshold {tresh}".format(size=sizes[i], tresh=fasttresh[j]))
                orb = cv.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=sizes[i], firstLevel=0, WTA_K=2, scoreType=ORB_HARRIS_SCORE , patchSize=sizes[i], fastThreshold=fasttresh[j])
                self.findKeyPoints(orb)
                self.matchKeyPointsBF() 
                self.filterMatches(plot=False, out=out)

                match_std, match_mean, match_median, match_spread, match_min, stats_plot = self.plotStats(sizes[i], fasttresh[j])
                effect_plot_index.append(index)
                effect_plot_settings.append((sizes[i], fasttresh[j]))
                effect_plot_match_mean.append(match_mean)
                effect_plot_match_min.append(match_min)
                if best_min < match_min:
                    best_std = match_std
                    best_mean = match_mean
                    best_median = match_median
                    best_spread = match_spread
                    best_min = match_min
                    best_patchsize = sizes[i]
                    best_fasttresh = fasttresh[j]
                    best_plt = stats_plot


                if i == (len(sizes)-1) and j == (len(fasttresh)-1):
                    best_plt.savefig(self.stats_folder + self.save_extension + ".png")
                    self.writeStatsToFile(best_std, best_mean, best_median, best_min, best_spread, best_patchsize, best_fasttresh)
                    settings_plot = self.plotEffectSettings(effect_plot_settings, effect_plot_match_mean, effect_plot_index, effect_plot_match_min)
                    settings_plot.savefig(self.stats_folder + self.save_extension + "_settings.png")
                    self.removeNonBestVideos(best_patchsize, best_fasttresh)
                self.track_error = 0
                self.keypoints = []
                self.descriptor = []
                self.matches = []
                self.matches_good = []
                plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''This script reads the dataset images or rosbag files and outputs the matches over time''')
    parser.add_argument('dataset', help='euroc, flourish, rosario, kitti, 4seasons, own')
    parser.add_argument('--source', help='specify source bag file or folder', default="")
    parser.add_argument('--equalize', help='Histogram equalization, True or False', default=False)
    parser.add_argument('--ds_fps', help='Downsample fps to equalize evaluation between datasets, True or False', default=False)
    parser.add_argument('--ds_resolution', help='Downsample resolution to equalize evaluation between datasets, True or False', default=False)
    parser.add_argument('--save_video', help='Save video with statistics', default=False)
    #args = parser.parse_args()
    args = parser.parse_args(["euroc", "--source", "/media/meltem/moo/Meltem/Thesis/Datasets/EuRoC/MH01/mav0/cam0/data", "--ds_fps", "False", "--save_video", "True"])
    object = ORB(args.dataset, args.source, args.equalize, args.ds_fps, args.ds_resolution, args.save_video)
    print("Settings set to equalize: {equalize}, downsample_fps: {ds_fps}, downsample_image: {ds_img}, save_video: {savevid}".format(equalize = args.equalize, ds_fps=args.ds_fps, ds_img=args.ds_resolution, savevid=args.save_video))
    object.main()

    #scatterplot met effecten van verschillende orb settings maken.
    # plots maken met behaviour vana mount of tracking failures in ORB-SLAM vs statistics van de dataset om te kijken welke een indicatie is van
    # tracking failure (mean, min, spread?)
    #performance metric: hoe vaak matches onder 20 komen.
    #own data with distorted view?
    # use frequentist approach, independent t-test (check conditions first; Independence of observations, Homogeneity of variance, Normality of data)