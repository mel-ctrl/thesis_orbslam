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
        self.histo_folder = "/home/meltem/thesis_orbslam/experiments/histograms/"
        self.statsfile =  "/home/meltem/thesis_orbslam/experiments/stats.yaml"
        self.save_stats_folder = self.stats_folder + self.dataset
        self.save_hist_folder = self.histo_folder + self.dataset
        self.blur = []
        self.flow = []
        self.frame_stats = []
        self.target_fps = 10
        self.target_width, self.target_height = 672, 376
        self.ds_fps = bool(ds_fps)
        self.ds_resolution = bool(ds_resolution)
        self.track_error = 0
        self.save_video = bool(save_video)
        self.out = None
        
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

        video_dir = self.stats_folder + "videos/" + self.save_extension
        self.frame_stats_file = self.stats_folder + self.save_extension + "_stats.csv"
        if not os.path.exists(self.save_stats_folder):
            os.makedirs(self.save_stats_folder)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if os.path.exists(self.frame_stats_file):
            os.remove(self.frame_stats_file)
        if not os.path.exists(self.frame_stats_file):
            header = ['patchsize', 'fasttresh', 'scalefactor', 'n_levels', 'track_error', 'features', 'total_matches', 'good_matches', 'blur', 'flow']
            with open(self.frame_stats_file, 'w') as statsfile:
                writer = csv.writer(statsfile)
                writer.writerow(header)
            statsfile.close()


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


    def filterMatches(self, plot, out, patchsize, fasttresh, scalefactor, n_levels):
        for image in range(len(self.matches)):
            matches_good = []
            matchesMask = None
            if(self.matches[image]):
                matchesMask = [[0,0] for i in range(len(self.matches[image]))]
                for i, pair in enumerate(self.matches[image]):
                    try:
                        m,n = pair
                        if m.distance < 0.75*n.distance:
                            if 'matchesMask' in locals():
                                matchesMask[i]=[1,0]
                            matches_good.append(m)
                    except(ValueError):
                        pass
            self.matches_good.append(matches_good)
            self.trackFrameStats(patchsize, fasttresh, scalefactor, n_levels, len(self.keypoints[image]), len(self.matches[image]), len(matches_good), self.blur[image], self.flow[image])

            if plot or self.save_video:
                draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (0,0,255),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
                img = cv.drawMatchesKnn(self.images[image],self.keypoints[image],self.images[image+1],self.keypoints[image+1],self.matches[image],None,**draw_params)
                if plot==True:
                    plt.figure(2)          
                    plt.imshow(img)
                    plt.show()
                if self.save_video:
                    patchsize = self.frame_stats[-1][0]
                    fasttresh = self.frame_stats[-1][1]
                    total_matches = self.frame_stats[-1][5]
                    good_matches = self.frame_stats[-1][6]
                    blur = self.frame_stats[-1][7]
                    flow = self.frame_stats[-1][8]

                    cv.rectangle(img,(0, int(4*img.shape[0]/5)),(int(img.shape[1]/5),int(img.shape[0])),(0,0,0),-1)
                    font = cv.FONT_HERSHEY_SIMPLEX 
                    cv.putText(img, "patch: {patchsize}, ftresh: {fasttresh}, scale: {scale}, levels: {nlevels}".format(patchsize = str(patchsize), fasttresh = str(fasttresh), scale=scalefactor, nlevels=n_levels),(0 , int(4*img.shape[0]/5+10)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                    cv.putText(img, "track error: {track_error}".format(track_error = str(self.track_error)),(0 , int(4*img.shape[0]/5+30)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                    cv.putText(img, "total matches: {total_matches} good matches: {good_matches}".format(total_matches=str(total_matches), good_matches=str(good_matches)),(0 , int(4*img.shape[0]/5+50)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                    cv.putText(img, "blur: {blur} flow: {flow}".format(blur=str(round(blur)), flow=str(round(flow))),(0 , int(4*img.shape[0]/5+70)), font, 0.4,(255,255,255),1,cv.LINE_AA)

                    out.write(img)


        if len(self.matches_good) < 20:
            self.track_error += 1
        
        if self.save_video:
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

        features = []
        xint = range(0, len(self.keypoints))
        for i in xint:
            features.append(len(self.keypoints[i]))

        features_std = round(np.std(features),1)
        features_mean = round(np.mean(features),1)
        features_median = np.median(features)
        features_spread= np.max(features)-np.min(features)
        features_min = np.min(features)
        

        data_dict = {'track_error': int(self.track_error), 'match_mean': float(match_mean), 'match_std': float(match_std), 'match_median': float(match_median), 'match_minimum': int(match_minimum), \
            'match_spread': int(match_spread), 'patchsize': int(patchsize), 'fasttresh': int(fasttresh), 'blur_mean': float(blur_mean), 'blur_std': float(blur_std), 'blur_median': float(blur_median), \
                'blur_maximum': float(blur_maximum), 'blur_spread': float(blur_spread), 'flow_mean': float(flow_mean), 'flow_std': float(flow_std), 'flow_median': float(flow_median), 'flow_maximum': float(flow_maximum), 'flow_spread': float(flow_spread), \
                 'features_mean': float(features_mean), 'features_std': float(features_std), 'features_median': float(features_median), 'features_minimum': int(features_min), 'features_spread': int(features_spread)}
        data = {self.save_extension : data_dict}
        with open(self.statsfile, "a") as file:
            yaml.dump(data, file)



    def plotStats(self, patchsize, fasttresh, n_levels, scalefactor):
        stats_plot = plt.figure(1)
        y = []
        xint = range(0, len(self.matches_good))
        for i in xint:
            y.append(len(self.matches_good[i]))
        plt.plot(xint, y)
        plt.axhline(np.mean(y), linestyle='--')

        std = round(np.std(y),1)
        mean = round(np.mean(y),1)
        median = np.median(y)
        spread = np.max(y)-np.min(y)
        min = np.min(y)
        plt.xlabel("Frame")
        plt.ylabel("Matches")
        plt.title("std: {std}, mean: {mean}, median: {median}, \n minimum: {min}, spread: {spread}, \n patchsize: {patchsize}, fasttresh: {fasttresh}, nlevels: {nlevels}, scalefactor: {scalefactor}".format(std=std, mean=mean, median=median, min=min, spread = spread, patchsize=patchsize, fasttresh=fasttresh, nlevels=n_levels, scalefactor=scalefactor))
        plt.tight_layout()
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
        if self.ds_resolution == True:
            img = cv.resize(img, (self.target_width, self.target_height))
        self.images.append(img)
        self.blur.append(self.variance_of_laplacian(img))
    
    def removeNonBestVideos(self, patchsize, fasttresh):
        folder = self.stats_folder + "videos/" + self.save_extension
        for file in os.listdir(folder):
            if (str(patchsize) + "_" + str(fasttresh) + ".avi") in file:
                continue
            else:
                os.remove(folder + "/" + file)

    def trackFrameStats(self, patchsize, fasttresh, scalefactor, n_levels, features, matches, matches_good, blur, flow):
        data = [patchsize, fasttresh, scalefactor, n_levels, self.track_error, features, matches, matches_good, blur, flow]
        self.frame_stats.append(data)

    
    def writeFrameStats(self, best_patchsize, best_treshold, best_scale, best_levels):
        with open(self.frame_stats_file, 'a') as frame_statsfile:
            writer = csv.writer(frame_statsfile)
            for row in self.frame_stats:
                if row[0] == (best_patchsize) and row[1] == (best_treshold) and row[2] == (best_scale) and row[3] == (best_levels):
                    writer.writerow(row)
        frame_statsfile.close()


    def main(self):
        print("Dataset: {dataset}".format(dataset=self.save_extension))
        img_idx = 0
        if self.dataset == "flourish" or self.dataset == "rosario" or self.dataset == "own":
            bag = rosbag.Bag(self.source, "r")
            bridge = CvBridge()
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img is not None and len(img) != 0:
                    if self.ds_fps and (self.fps != self.target_fps):
                        img_idx +=1
                        if img_idx % (self.fps/(self.fps-self.target_fps)) != 0:
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

        print('Calculating optical flow between all images')
        self.calculateFlow()

        sizes = [6, 12, 24, 48]
        fasttresh = [5, 10, 20, 40]
        scale_factor = [1.1, 1.2, 1.3, 1.4]
        n_levels = [4, 8, 12, 16]
        n_features = 2000
        best_std = 0
        best_mean = 0
        best_median = 0
        best_spread = 0
        best_min = 0
        best_patchsize = 0
        best_fasttresh = 0
        best_plt = None
        index = 0

        for size_idx in range(len(sizes)):
            for fasttresh_idx in range(len(fasttresh)):
                for scalefactor_idx in range(len(scale_factor)):
                    for levels_idx in range(len(n_levels)):
                        if self.save_video:
                            fourcc = cv.VideoWriter_fourcc(*'MJPG')
                            if self.ds_resolution == True:
                                self.out = cv.VideoWriter((self.stats_folder + "videos/" + self.save_extension + "/" + str(sizes[size_idx]) + "_" + str(fasttresh[fasttresh_idx]) + ".avi"), fourcc, self.target_fps, (self.target_width*2, self.target_height))
                            else:
                                self.out = cv.VideoWriter((self.stats_folder + "videos/" + self.save_extension + "/" + str(sizes[size_idx]) + "_" + str(fasttresh[fasttresh_idx]) + ".avi"), fourcc, self.target_fps, (img.shape[1]*2, img.shape[0]))
                        index+=1
                        print("Matching features for patch size {size} and fast treshold {tresh}".format(size=sizes[size_idx], tresh=fasttresh[fasttresh_idx]))
                        orb = cv.ORB_create(nfeatures=n_features, scaleFactor=scale_factor[scalefactor_idx], nlevels=n_levels[levels_idx], edgeThreshold=sizes[size_idx], firstLevel=0, WTA_K=2, scoreType=ORB_HARRIS_SCORE , patchSize=sizes[size_idx], fastThreshold=fasttresh[fasttresh_idx])
                        self.findKeyPoints(orb)
                        self.matchKeyPointsBF() 
                        self.filterMatches(plot=False, out=self.out, patchsize=sizes[size_idx], fasttresh=fasttresh[fasttresh_idx], scalefactor=scale_factor[scalefactor_idx], n_levels=n_levels[levels_idx])

                        match_std, match_mean, match_median, match_spread, match_min, stats_plot = self.plotStats(sizes[size_idx], fasttresh[fasttresh_idx], n_levels[levels_idx], scale_factor[scalefactor_idx])
                        if best_min < match_min:
                            best_std = match_std
                            best_mean = match_mean
                            best_median = match_median
                            best_spread = match_spread
                            best_min = match_min
                            best_patchsize = sizes[size_idx]
                            best_fasttresh = fasttresh[fasttresh_idx]
                            best_scale = scale_factor[scalefactor_idx]
                            best_levels = n_levels[levels_idx]
                            best_plt = stats_plot

                        if size_idx == (len(sizes)-1) and levels_idx == (len(n_levels)-1):
                            best_plt.savefig(self.stats_folder + self.save_extension + ".png")
                            self.writeStatsToFile(best_std, best_mean, best_median, best_min, best_spread, best_patchsize, best_fasttresh)
                            self.writeFrameStats(best_patchsize, best_fasttresh,  best_scale, best_levels)
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
    args = parser.parse_args()
    #args = parser.parse_args(["kitti", "--source", "/home/meltem/imow_line/visionTeam/Meltem/Datasets/kitti/data_odometry_color/dataset/sequences/00/image_2", "--ds_fps", "False", "--ds_resolution", "False", "--save_video", "False"])
    object = ORB(args.dataset, args.source, args.equalize, args.ds_fps, args.ds_resolution, args.save_video)
    print("Settings set to equalize: {equalize}, downsample_fps: {ds_fps}, downsample_image: {ds_img}, save_video: {savevid}".format(equalize = args.equalize, ds_fps=args.ds_fps, ds_img=args.ds_resolution, savevid=args.save_video))
    object.main()
    #scatterplot met effecten van verschillende orb settings maken.
    # plots maken met behaviour vana mount of tracking failures in ORB-SLAM vs statistics van de dataset om te kijken welke een indicatie is van
    # tracking failure (mean, min, spread?)
    #performance metric: hoe vaak matches onder 20 komen.
    #own data with distorted view?
    # use frequentist approach, independent t-test (check conditions first; Independence of observations, Homogeneity of variance, Normality of data)