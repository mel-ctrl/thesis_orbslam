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
        self.image_name = []
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
        self.contrast = []
        self.frame_stats = []
        self.target_fps = 10
        self.target_width, self.target_height = 672, 376
        self.ds_fps = bool(ds_fps)
        self.ds_resolution = bool(ds_resolution)
        self.track_error = 0
        self.save_video = bool(save_video)
        self.out = None
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
        elif self.dataset == "slam":
            self.save_extension = self.dataset + "/" + (self.source.split('/')[-1])
            self.fps = 10
        else:
            self.save_extension = "test"
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
            #header = ['patchsize', 'fasttresh', 'scalefactor', 'n levels', 'track error', 'features', 'total matches', 'inliers', 'orbslam features', 'orbslam matches', 'blur', 'flow', 'contrast', 'average intensity', 'percentage overexposure', 'percentage underexposure']
            header = ['patchsize', 'fasttresh', 'scalefactor', 'n_levels', 'track_error', 'features', 'total_matches', 'inliers', 'blur', 'flow', 'contrast', 'average intensity', 'percentage white pixels', 'percentage black pixels']

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
            else:
                self.keypoints.append([])
                self.descriptor.append([])

    def drawKeyPoints(self):
        keypoint_plot = plt.figure(3)
        img = cv.drawKeypoints(self.images[0], self.keypoints[0], None, color=(0,255,0), flags=0)
        plt.imshow(img)
        keypoint_plot.show()

    def matchKeyPointsBF(self):
        bf = cv.BFMatcher()
        
        #for i in range(0,len(self.images)-1,2):
        for i in range(0, len(self.images)-1):
            try:
                matches = bf.knnMatch(self.descriptor[i], self.descriptor[i+1], k=2)
            except:
                matches = []
                
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


    def filterMatches(self, plot, out, patchsize, fasttresh, scalefactor, n_levels):
        pair_list = [*range(0, len(self.images), 2)]
        for image in range(len(self.matches)):
            #img_pair = pair_list[image]
            matches_good = []
            #matchesMask = None
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
            #homo_mat, matchesMask = self.findHomography(matches_good, img_pair)
            homo_mat, matchesMask = self.findHomography(matches_good, image)
            self.matches_good.append(matches_good)
            self.inliers.append(matchesMask.count(1))
            #orbslam_features = (self.image_name[image].split('_')[-2])#.split('.')[0]
            #orbslam_matches = (self.image_name[image].split('_')[-3])
            flattened_img = np.asarray(self.images[image]).flatten()
            contrast = round(np.std((flattened_img-np.min(flattened_img))/(np.max(flattened_img-np.min(flattened_img)))),2)
            self.contrast.append(contrast)
            intensity_unique, intensity_counts = np.unique(self.images[image], return_counts=True)
            count_dict = dict(zip(intensity_unique, intensity_counts))
            sum_black = 0
            sum_white = 0

            for i in range(0,6):
                if i in count_dict:
                    sum_black = sum_black + count_dict[i]
            for i in range(250, 256):
                if i in count_dict:
                    sum_white = sum_white + count_dict[i]

            perc_overexposure =sum_white/(self.images[image].shape[0]*self.images[image].shape[1])*100
            perc_underexposure = sum_black/(self.images[image].shape[0]*self.images[image].shape[1])*100

            average_intensity = np.mean(self.images[image])
            #self.trackFrameStats(patchsize, fasttresh, scalefactor, n_levels, len(self.keypoints[image]), len(self.matches[image]), matchesMask.count(1), orbslam_features, orbslam_matches, self.blur[image], self.flow[image], contrast, average_intensity, perc_overexposure, perc_underexposure)
            self.trackFrameStats(patchsize, fasttresh, scalefactor, n_levels, len(self.keypoints[image]), len(self.matches[image]), matchesMask.count(1), self.blur[image], self.flow[image], contrast, average_intensity, perc_overexposure, perc_underexposure)

            #self.savePlotAndVideo(plot, out, patchsize, fasttresh, scalefactor, n_levels, matchesMask, img_pair, image, matches_good)
            self.savePlotAndVideo(plot, out, patchsize, fasttresh, scalefactor, n_levels, matchesMask, pair_list, image,matches_good)

        if (matchesMask.count(1)) < 20:
            self.track_error += 1
        if self.save_video:
            out.release()
            cv.destroyAllWindows()

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


        

    def savePlotAndVideo(self, plot, out, patchsize, fasttresh, scalefactor, n_levels, matchesMask, pair, image, matches_good):
        if plot==True or self.save_video:
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (0,0,255),
                   matchesMask = matchesMask, # draw only inliers
                   flags = cv.DrawMatchesFlags_DEFAULT)
            
            #img1, img2 = self.images[pair], self.images[pair+1]
            #img = cv.drawMatches(img1,self.keypoints[pair],img2,self.keypoints[pair+1],matches_good,None,**draw_params)

            img1, img2 = self.images[image], self.images[image+1]
            img = cv.drawMatches(img1,self.keypoints[image],img2,self.keypoints[image+1],matches_good,None,**draw_params)
            if plot==True:
                plt.figure(2)          
                plt.imshow(img)
                plt.show()
            
            if self.save_video:
                patchsize = self.frame_stats[-1][0]
                fasttresh = self.frame_stats[-1][1]
                total_matches = self.frame_stats[-1][6]
                inliers = self.frame_stats[-1][7]
                blur = self.frame_stats[-1][8]
                flow = self.frame_stats[-1][9]
                contrast = self.frame_stats[-1][10]

                #blur = self.frame_stats[-1][10]
                #flow = self.frame_stats[-1][11]
                #contrast = self.frame_stats[-1][12]

                cv.rectangle(img,(0, int(4*img.shape[0]/5)),(int(img.shape[1]/5),int(img.shape[0])),(0,0,0),-1)
                font = cv.FONT_HERSHEY_SIMPLEX 
                cv.putText(img, "patch: {patchsize}, ftresh: {fasttresh}, scale: {scale}, levels: {nlevels}".format(patchsize = str(patchsize), fasttresh = str(fasttresh), scale=scalefactor, nlevels=n_levels),(0 , int(4*img.shape[0]/5+10)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                cv.putText(img, "track error: {track_error}".format(track_error = str(self.track_error)),(0 , int(4*img.shape[0]/5+30)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                cv.putText(img, "total matches: {total_matches} inliers: {inliers}".format(total_matches=str(total_matches), inliers=str(inliers)),(0 , int(4*img.shape[0]/5+50)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                cv.putText(img, "blur: {blur} flow: {flow} contrast: {contrast}".format(blur=str(round(blur)), flow=str(round(flow)), contrast=str(contrast)),(0 , int(4*img.shape[0]/5+70)), font, 0.4,(255,255,255),1,cv.LINE_AA)
                out.write(img)


    def calculateFlow(self):
        for i in range(len(self.images)-1):
            flow = cv.calcOpticalFlowFarneback(self.images[i], self.images[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
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

        contrast_std = round(np.std(self.contrast),2)
        contrast_mean = round(np.mean(self.contrast),2)
        contrast_median =np.median(self.contrast)
        contrast_maximum = np.max(self.contrast)
        contrast_spread = round((np.max(self.contrast)-np.min(self.contrast)),2)

        data_dict = {'track_error': int(self.track_error), 'match_mean': float(match_mean), 'match_std': float(match_std), 'match_median': float(match_median), 'match_minimum': int(match_minimum), \
            'match_spread': int(match_spread), 'patchsize': int(patchsize), 'fasttresh': int(fasttresh), 'blur_mean': float(blur_mean), 'blur_std': float(blur_std), 'blur_median': float(blur_median), \
                'blur_maximum': float(blur_maximum), 'blur_spread': float(blur_spread), 'flow_mean': float(flow_mean), 'flow_std': float(flow_std), 'flow_median': float(flow_median), 'flow_maximum': float(flow_maximum), 'flow_spread': float(flow_spread), \
                 'features_mean': float(features_mean), 'features_std': float(features_std), 'features_median': float(features_median), 'features_minimum': int(features_min), 'features_spread': int(features_spread), 'contrast_mean':float(contrast_mean), \
                    'contrast_std':float(contrast_std), 'contrast_median':float(contrast_median), 'contrast_maximum':float(contrast_maximum), 'contrast_spread':float(contrast_spread)}
        data = {self.save_extension : data_dict}
        with open(self.statsfile, "a") as file:
            yaml.dump(data, file)



    def plotStats(self, patchsize, fasttresh, n_levels, scalefactor):
        stats_plot = plt.figure(1)
        xint = range(0, len(self.matches_good))
        y = self.inliers
        '''
        y = []
        for i in xint:
            y.append(len(self.matches_good[i]))
        plt.plot(xint, y)
        '''
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

    def addImages(self, img, filename):
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if self.equalize == True:
            img = self.equalizeHist(img)
        if self.ds_resolution == True:
            img = cv.resize(img, (self.target_width, self.target_height))
        self.images.append(img)
        self.image_name.append(filename)
        self.blur.append(self.variance_of_laplacian(img))
    
    def removeNonBestVideos(self, patchsize, fasttresh):
        folder = self.stats_folder + "videos/" + self.save_extension
        for file in os.listdir(folder):
            if (str(patchsize) + "_" + str(fasttresh) + ".avi") in file:
                continue
            else:
                os.remove(folder + "/" + file)

    def trackFrameStats(self, patchsize, fasttresh, scalefactor, n_levels, features, matches, inliers, blur, flow, contrast, average_intensity, perc_overexposure, perc_underexposure):
    #def trackFrameStats(self, patchsize, fasttresh, scalefactor, n_levels, features, matches, inliers,orbslam_features,  orbslam_matches, blur, flow, contrast,  average_intensity, perc_overexposure, perc_underexposure):
        #data = [patchsize, fasttresh, scalefactor, n_levels, self.track_error, features, matches, inliers, orbslam_features, orbslam_matches, blur, flow, contrast, average_intensity, perc_overexposure, perc_underexposure]
        data = [patchsize, fasttresh, scalefactor, n_levels, self.track_error, features, matches, inliers, blur, flow, contrast, average_intensity, perc_overexposure, perc_underexposure]
        self.frame_stats.append(data)

    
    def writeFrameStats(self, best_patchsize, best_treshold, best_scale, best_levels):
        with open(self.frame_stats_file, 'a') as frame_statsfile:
            writer = csv.writer(frame_statsfile)
            for count, row in enumerate(self.frame_stats):
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
                    filename = str(img_idx)
                    if self.ds_fps and (self.fps != self.target_fps):
                        img_idx +=1
                        if img_idx % (self.fps/(self.fps-self.target_fps)) != 0:
                            self.addImages(img, filename)
                    else:
                        self.addImages(img,filename)
            bag.close()

        else:
            for filename in sorted(os.listdir(self.source)):
                img = cv.imread(os.path.join(self.source,filename))
                if img is not None and len(img) != 0:
                    if self.ds_fps and (self.fps != self.target_fps):
                        img_idx +=1
                        if img_idx % (self.fps/(self.fps-self.target_fps)) != 0:
                            self.addImages(img, filename)
                    else:
                        self.addImages(img, filename)
        print("Read all images")

        print('Calculating optical flow between all images')
        self.calculateFlow()

        sizes = [48] #[31] [6, 12, 24, 48]
        fasttresh = [5] #[5, 10, 20, 40]
        scale_factor = [1.2]#[1.1, 1.2, 1.3, 1.4]
        n_levels = [8] #[4, 8, 12, 16]
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
            for scalefactor_idx in range(len(scale_factor)):
                for levels_idx in range(len(n_levels)):
                    for fasttresh_idx in range(len(fasttresh)):
                        if self.save_video:
                            fourcc = cv.VideoWriter_fourcc(*'MJPG')
                            if self.ds_resolution == True:
                                self.out = cv.VideoWriter((self.stats_folder + "videos/" + self.save_extension + "/" + str(sizes[size_idx]) + "_" + str(fasttresh[fasttresh_idx]) + ".avi"), fourcc, self.target_fps, (self.target_width*2, self.target_height))
                            else:
                                self.out = cv.VideoWriter((self.stats_folder + "videos/" + self.save_extension + "/" + str(sizes[size_idx]) + "_" + str(fasttresh[fasttresh_idx]) + ".avi"), fourcc, self.target_fps, (img.shape[1]*2, img.shape[0]))
                        index+=1
                        print("Matching features for patch size {size}, fast treshold {tresh}, scale factor {scalefactor}, nlevels {nlevels}".format(size=sizes[size_idx], tresh=fasttresh[fasttresh_idx], scalefactor=scale_factor[scalefactor_idx], nlevels=n_levels[levels_idx]))
                        orb = cv.ORB_create(nfeatures=n_features, scaleFactor=scale_factor[scalefactor_idx], nlevels=n_levels[levels_idx], edgeThreshold=sizes[size_idx], firstLevel=0, WTA_K=2, scoreType=ORB_HARRIS_SCORE , patchSize=sizes[size_idx], fastThreshold=fasttresh[fasttresh_idx])
                        #change for edgethresh = sizes[size_idx]
                        self.findKeyPoints(orb)
                        self.matchKeyPointsBF() 
                        self.filterMatches(plot=False, out=self.out, patchsize=sizes[size_idx], fasttresh=fasttresh[fasttresh_idx], scalefactor=scale_factor[scalefactor_idx], n_levels=n_levels[levels_idx])

                        match_std, match_mean, match_median, match_spread, match_min, stats_plot = self.plotStats(sizes[size_idx], fasttresh[fasttresh_idx], n_levels[levels_idx], scale_factor[scalefactor_idx])

                        if best_min <= match_min:
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

                        if size_idx == (len(sizes)-1) and scalefactor_idx==(len(scale_factor)-1) and levels_idx==(len(n_levels)-1) and fasttresh_idx==(len(fasttresh)-1):
                            best_plt.savefig(self.stats_folder + self.save_extension + ".png")
                            self.writeStatsToFile(best_std, best_mean, best_median, best_min, best_spread, best_patchsize, best_fasttresh)
                            self.writeFrameStats(best_patchsize, best_fasttresh, best_scale, best_levels)
                            self.removeNonBestVideos(best_patchsize, best_fasttresh)
                        
                        self.track_error = 0
                        self.keypoints = []
                        self.descriptor = []
                        self.matches = []
                        self.matches_good = []
                        self.inliers = []
                        self.contrast = []
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
    #args = parser.parse_args(["euroc", "--source", "/media/meltem/moo/EuRoC/MH01/mav0/cam0/data", "--ds_fps", "False", "--ds_resolution", "False", "--save_video", "True", "--equalize", "False"])
    #args = parser.parse_args(["slam", "--source", "/home/meltem/thesis_orbslam/imgs_failed_match_juno/new_settings/imgs_failed_match_juno_den_boer_sunny", "--ds_fps", "False", "--save_video", "True"])
    
    object = ORB(args.dataset, args.source, args.equalize, args.ds_fps, args.ds_resolution, args.save_video)
    print("Settings set to equalize: {equalize}, downsample_fps: {ds_fps}, downsample_image: {ds_img}, save_video: {savevid}".format(equalize = args.equalize, ds_fps=args.ds_fps, ds_img=args.ds_resolution, savevid=args.save_video))
    object.main()