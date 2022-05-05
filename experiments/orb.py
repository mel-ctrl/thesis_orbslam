from cv2 import ORB_HARRIS_SCORE, imshow
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import yaml
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ORB:
    def __init__(self, source_dir, settings_path, dataset, bag_file):
        self.source_dir = source_dir
        self.images = []
        self.orb = cv.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8, edgeThreshold=19, firstLevel=0, WTA_K=2, scoreType=ORB_HARRIS_SCORE , patchSize=31, fastThreshold=20)
        self.keypoints = []
        self.descriptor = []
        self.matches = []
        self.matches_good = []
        self.dataset = dataset
        self.bag_file = bag_file
        self.settings_path = settings_path
        if self.dataset == "euroc":
            self.mtx, self.dist = self.readSettings()
        elif self.dataset == "flourish":
            self.image_topic = "/sensor/camera/vi_sensor/left/image_raw"
            self.mtx, self.dist = self.readSettings()
        elif self.dataset == "rosario":
            self.image_topic = "/stereo/left/image_raw"
            self.mtx, self.dist = self.readSettings()
                

    def readSettings(self):
        if self.dataset == "flourish" or self.dataset == "rosario":
            if self.dataset == "flourish":
                camera_info_topic = "/sensor/camera/vi_sensor/left/camera_info"
            elif self.dataset == "rosario":
                camera_info_topic = "/stereo/left/camera_info"
            bag = rosbag.Bag(self.bag_file, "r")
            i = 0
            for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
                bag.read_messages(topics=[camera_info_topic])
                intrinsics = msg.K
                distortion_coeff = msg.D
                mtx = np.array([[intrinsics[0], intrinsics[1], intrinsics[2]], [intrinsics[3], intrinsics[4], intrinsics[5]],[intrinsics[6], intrinsics[7], intrinsics[8]]])
                dist = np.asarray(distortion_coeff)
                i+=1
                if i == 1:
                    break
            return mtx, dist
            

        else:
            with open(self.settings_path) as file: 
                settings = yaml.full_load(file)
                if self.dataset == "euroc":
                    intrinsics = settings["intrinsics"]
                    distortion_coeff = settings["distortion_coefficients"]


                mtx = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]],[0, 0, 1]])
                return mtx, np.asarray(distortion_coeff)

    def findKeyPoints(self):
        for img in self.images:
            kp, des = self.orb.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptor.append(des)

    def drawKeyPoints(self):
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
                    print("bug happened")
            
            self.matches.append(matches)


    def drawMatches(self, plot):
        for image in range(len(self.images)-1):
            matches_good = []
            matchesMask = [[0,0] for i in range(len(self.matches[image]))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(self.matches[image]):
                #print(image, i, len(self.matches[image]))
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
                    matches_good.append(m)

            self.matches_good.append(matches_good)

            if plot==True:           
                draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)

                img = cv.drawMatchesKnn(self.images[image],self.keypoints[image],self.images[image+1],self.keypoints[image+1],self.matches[image],None,**draw_params)
                plt.imshow(img)
                plt.show()

    def undistortImg(self, img):
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        # undistort
        dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def plotStats(self):
        y = []
        xint = range(0, len(self.matches_good))
        for i in xint:
            y.append(len(self.matches_good[i]))
        plt.plot(xint, y)
        #e = y-np.mean(y)
        print(y)
        plt.axhline(np.mean(y), linestyle='--')
        #plt.errorbar(xint, y, e, linestyle='None', marker='^')
        plt.title("Amount of good matches per frame, std: {std}, \n mean: {mean}, median: {median}, minimum: {min}, spread: {spread}".format(std=np.std(y), mean=np.mean(y), median=np.median(y), min=min(y), spread = max(y)-min(y)))
        #plt.xticks(xint)
        plt.show()

        



    def main(self):
        if self.dataset == "flourish" or self.dataset == "rosario":
            bag = rosbag.Bag(self.bag_file, "r")
            bridge = CvBridge()
            i = 0
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if cv_img is not None:
                    img_undistorted = self.undistortImg(cv_img) 
                    self.images.append(img_undistorted)
                    i+=1
                    #if i == 6:
                    #    break
            bag.close()



        else:
            for filename in os.listdir(self.source_dir):
                img = cv.imread(os.path.join(self.source_dir,filename))
                if img is not None:
                    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    img_undistorted = self.undistortImg(img) 
                    self.images.append(img_undistorted)
                if len(self.images) == 5:
                    break
        print(len(self.images))
        self.findKeyPoints()
        #self.drawKeyPoints()
        #self.matchKeyPointsBF()
        self.matchKeyPointsFLANN()
        
        self.drawMatches(plot=False)
        self.plotStats()

if __name__ == "__main__":
    #-------------------------------ROSARIO-------------------------------------------------
    dataset = "rosario"
    bag_file = "/media/meltem/T7/Meltem/Thesis/Datasets/Rosario/sequence02.bag"
    source_dir = ""
    settings_path = ""


    #-------------------------------EUROC-------------------------------------------------
    #dataset = "euroc"
    #source_dir = "/media/meltem/T7/Meltem/Thesis/Datasets/EuRoC/MH01/mav0/cam0/data"
    #settings_path = "/media/meltem/T7/Meltem/Thesis/Datasets/EuRoC/MH01/mav0/cam0/sensor.yaml"

    #-------------------------------FLOURISH-------------------------------------------------
    
    #dataset= "flourish"
    #bag_file = "/media/meltem/T7/Meltem/Thesis/Datasets/Flourish/DatasetA.bag"
    #source_dir = ""
    #settings_path = ""

    object = ORB(source_dir, settings_path, dataset, bag_file)
    object.main()
