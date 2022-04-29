import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

class ORB:
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.images = []
        self.orb = cv.ORB_create()
        self.keypoints = []
        self.descriptor = []
        self.matches = []


    def findKeyPoints(self):
        for img in self.images:
            kp, des = self.orb.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptor.append(des)

            print(len(kp))


    def drawKeyPoints(self):
        img = cv.drawKeypoints(self.images[0], self.keypoints[0], None, color=(0,255,0), flags=0)
        plt.imshow(img)
        plt.show()

    def matchKeyPointsBF(self):
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        for i in range(len(self.images)-1):
            matches = bf.match(self.descriptor[i], self.descriptor[i+1])
            matches = sorted(matches, key = lambda x:x.distance)
            self.matches.append(matches)

    def drawMatches(self):
        img = cv.drawMatches(self.images[0],self.keypoints[0],self.images[1],self.keypoints[1],self.matches[0],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img)
        plt.show()


    def main(self):
        for filename in os.listdir(self.source_dir):
            img = cv.imread(os.path.join(self.source_dir,filename))
            if img is not None:
                self.images.append(img)
            if len(self.images) == 2:
                break

        self.findKeyPoints()
        self.drawKeyPoints()
        self.matchKeyPointsBF()
        self.drawMatches()


if __name__ == "__main__":
    source_dir = "/media/meltem/T7/Meltem/Thesis/Datasets/EuRoC_prepped/MH_01_easy/mav0/cam0/data/"
    object = ORB(source_dir)
    object.main()
