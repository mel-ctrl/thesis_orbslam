import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import csv
from scipy.spatial.transform import Rotation

from lib.visualization import plotting
from lib.visualization.video import play_trip
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VisualOdometry():
    def __init__(self, data_dir, dataset, poses_path, calib_paths):
        self.images_l = []
        self.images_r = []
        self.matches = 0
        self.inliers = 0
        if dataset == "kitti":
            self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt')
            self.gt_poses = self._load_poses('/media/meltem/moo/kitti/GT/02.txt')
            self.images_l = self._load_images(data_dir + '/image_0')
            self.images_r = self._load_images(data_dir + '/image_1')
        
        elif dataset == "own":
            den_boer_sunny_start_angle = 0.9 * np.pi #-1.84101874148744
            angle = den_boer_sunny_start_angle 
            self.K_l, self.P_l, self.D_l, self.R_l = self._load_calib_own(calib_paths[0])
            self.K_r, self.P_r, self.D_r, self.R_r  = self._load_calib_own(calib_paths[1])
            self.gt_poses = self._load_poses_own(poses_path, angle)
            

            self.image_topic_left = "/daheng_camera_manager/left/image_rect"
            self.image_topic_right = "/daheng_camera_manager/right/image_rect"

        elif dataset == "flourish":
            self.K_l = np.array([[468.7667304393543, 0.0, 368.7056252615736], [0.0, 468.0464195624461, 213.4293421094116], [0.0, 0.0, 1.0]])
            self.P_l = np.array([[453.00250474851873, 0.0, 368.8472023010254, 0.0], [0.0, 453.00250474851873, 220.84287071228027, 0.0], [0.0, 0.0, 1.0, 0.0]])
            self.D_l = np.array([-0.2842244675887264, 0.07805992922287622, -0.0009221559580732882, 0.000351441703602022, 0.0])
            self.R_l = np.array([[0.9999923083495567, -0.0030180494085867977, 0.0025049190590962205], [0.003021457941992808, 0.9999945132052477, -0.0013580689615490399], [-0.0025008065958932423, 0.0013656270233424583, 0.9999959405063618]])
            
            self.K_r = np.array([[470.4054651057866, 0.0, 373.609713379638], [0.0, 469.719654946929, 231.2805719016278], [0.0, 0.0, 1.0]])
            self.P_r = np.array([[453.00250474851873, 0.0, 368.8472023010254, -49.37568502792493], [0.0, 453.00250474851873, 220.84287071228027, 0.0], [0.0, 0.0, 1.0, 0.0]])
            self.D_r = np.array([-0.2857404065599007, 0.07960771615567709, -0.0004332477206773495, 0.0004782360449639518, 0.0])
            self.R_r = np.array([[0.9999853112261422, 0.0023132970523988146, 0.0049016312287895065], [-0.0023199702212165457, 0.9999963892643706, 0.0013561697511710828], [-0.004898476306807043, -0.001367521469186478, 0.9999870673238245]])
            self.gt_poses = self._load_poses_flourish('/media/meltem/moo/Flourish/GT/DatasetA_GT.txt')
            self.image_topic_left = "/sensor/camera/vi_sensor/left/image_raw"
            self.image_topic_right = "/sensor/camera/vi_sensor/right/image_raw"

        if dataset == "own" or dataset=="flourish":
            bag = rosbag.Bag(data_dir, "r")
            bridge = CvBridge()
            i = 0
            for topic, msg, t in bag.read_messages(topics=[self.image_topic_left, self.image_topic_right]):
                #print(msg.header.stamp)
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img is not None and len(img) != 0:
                    if topic == self.image_topic_left:
                        gray = self._gray_images(img, 'left')
                        self.images_l.append(gray)
                    elif topic == self.image_topic_right:
                        gray = self._gray_images(img, 'right')
                        self.images_r.append(gray)
                    i +=1
                #if i == 5:
                #    break
                
        block = 3 #11
        smooth = 0.0
        P1 = int(block * block * 8 * smooth)
        P2 = int(block * block * 32 * smooth)
        self.disparity = cv2.StereoSGBM_create(minDisparity=1, numDisparities=48, blockSize=block, P1=P1, P2=P2, speckleRange=0, speckleWindowSize=0)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]

        self.fastFeatures = cv2.FastFeatureDetector_create()
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=48, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=48, fastThreshold=5)
        #self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.05, nlevels=8, edgeThreshold=3, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=3, fastThreshold=1)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    def _gray_images(self, img, topic):
        if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.flip(img, 1)
        if topic == 'left':
            map1_x, map1_y = cv2.initUndistortRectifyMap(self.K_l, self.D_l, self.R_l, self.P_l, (img.shape[1], img.shape[0]), cv2.CV_32FC1)
            img = cv2.remap(img, map1_x, map1_y , cv2.INTER_CUBIC)
        elif topic == 'right':
            map2_x, map2_y = cv2.initUndistortRectifyMap(self.K_r, self.D_r, self.R_r, self.P_r, (img.shape[1], img.shape[0]), cv2.CV_32FC1)
            img = cv2.remap(img, map2_x, map2_y , cv2.INTER_CUBIC)

        return img

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_calib_own(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """

        with open(filepath, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            D = np.array(params["distortion_coefficients"]["data"]).reshape(1,5)
            P = np.array(params["projection_matrix"]["data"]).reshape(1,12)
            P = np.reshape(P, (3, 4))
            K = np.array(params["camera_matrix"]["data"]).reshape(1,9)
            K = np.reshape(K, (3, 3))
            R = np.array(params["rectification_matrix"]["data"]).reshape(1,9)
            R = np.reshape(R, (3,3))
        return K, P, D, R

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses


    @staticmethod
    def _load_poses_own(filepath, angle):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                t = np.fromstring(line, dtype=np.float64, sep=' ')
                t[1], t[2] = t[2], t[1]
                t = np.array(t[1:4]).reshape(3,)
                R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
                T = np.empty((4, 4))
                T[:3, :3] = R
                T[:3, 3] = t
                T[3, :] = [0, 0, 0, 1]
                poses.append(T)
        return poses

    @staticmethod
    def _load_poses_flourish(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                poseline = np.fromstring(line, dtype=np.float64, sep=' ')
                t = poseline[1:4]
                q_w = poseline[4:]
                q = np.append(q_w[1:], q_w[0])
                R = Rotation.from_quat(q_w)
                R = R.as_matrix()
                T = np.empty((4, 4))
                T[:3, :3] = R
                T[:3, 3] = t
                T[3, :] = [0, 0, 0, 1]
                poses.append(T)
        return poses


    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals




    def get_tiled_keypoints(self, img, tile_h, tile_w):
            """
            Splits the image into tiles and detects the 10 best keypoints in each tile

            Parameters
            ----------
            img (ndarray): The image to find keypoints in. Shape (height, width)
            tile_h (int): The tile height
            tile_w (int): The tile width

            Returns
            -------
            kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
            """
            def get_kps(x, y):
                # Get the image tile
                impatch = img[y:y + tile_h, x:x + tile_w]

                # Detect keypoints
                keypoints = self.fastFeatures.detect(impatch)

                # Correct the coordinate for the point
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                # Get the 10 best keypoints
                if len(keypoints) > 10:
                    keypoints = sorted(keypoints, key=lambda x: -x.response)
                    return keypoints[:10]
                return keypoints
            # Get the image height and width
            h, w, *_ = img.shape

            # Get the keypoints for each of the tiles
            kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

            # Flatten the keypoint list
            kp_list_flatten = np.concatenate(kp_list)
            #img2 = cv2.drawKeypoints(img, kp_list_flatten, None, color=(0,255,0), flags=0)
            #plt.imshow(img2), plt.show()
            return kp_list_flatten

    def get_tiled_keypoints2(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints, descriptors = self.orb.detectAndCompute(impatch, None)
            #keypoints = self.fastFeatures.detect(impatch)
            # Correct the coordinate for the point
            
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                sortedpair = sorted(zip(keypoints, descriptors), key=lambda pair: -pair[0].response)
                bestx = sortedpair[:10]
                keypoints, descriptors = zip(*bestx)
                return keypoints, descriptors
            return keypoints, descriptors
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y)[0] for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
        ds_list = [get_kps(x, y)[1] for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        kp_list =  list(filter(lambda item: item is not None, kp_list))
        ds_list =  list(filter(lambda item: item is not None, ds_list))
        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        ds_list_flatten = np.concatenate(ds_list)

        #img2 = cv2.drawKeypoints(img, kp_list_flatten, None, color=(0,255,0), flags=0)
        #plt.imshow(img2), plt.show()
        return kp_list_flatten, ds_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=8):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        
        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])
        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def track_keypointsbymatch(self, img1, img2, kp1, kp2_l, ds1_l, ds_2_l, max_error=100):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        trackpoints1 = []
        trackpoints2 = []
        matches_good = []

        matches = self.bf.knnMatch(ds1_l, ds_2_l, k=2)
        for i, match_pair in enumerate(matches):
            try:
                m,n = match_pair 
                if m.distance < 0.75*n.distance:
                    if m.distance < max_error:
                        matchingquerypoint = kp1[m.queryIdx]
                        matchingtrainpoint = kp2_l[m.trainIdx]
                        matches_good.append(m)
                        #trackpoints1.append(matchingquerypoint.pt)
                        #trackpoints2.append(matchingtrainpoint.pt)               
            except(ValueError):
                pass
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2_l,matches_good,None,flags=cv2.DrawMatchesFlags_DEFAULT)
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2_l[m.trainIdx].pt for m in matches_good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,20.0)
        matchesMask = mask.ravel().tolist()
        matchesMask = np.array(matchesMask).astype(np.bool)
        matches_good = np.array(matches_good)
        inliers = matches_good[matchesMask]
        trackpoints1 = [kp1[x.queryIdx].pt for x in inliers]
        trackpoints2 = [kp2_l[x.trainIdx].pt for x in inliers]

        self.matches = len(matches_good)
        self.inliers = len(inliers)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            singlePointColor = (0,0,255),
            matchesMask = matchesMask.tolist(), # draw only inliers
            flags = cv2.DrawMatchesFlags_DEFAULT)
        
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2_l,matches_good.tolist(),None,**draw_params)
        #plt.imshow(img3),plt.show()

        return np.array(trackpoints1), np.array(trackpoints2)
    
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=1.0, max_disp=48.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=500):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 10

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=2000,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]
        img1_r, img2_r = self.images_r[i - 1:i + 1]


        # Get teh tiled keypoints
        #kp1_l = self.get_tiled_keypoints(img1_l, 10, 20) #10, 20
        #kp2_l, ds2_l = self.get_tiled_keypoints(img2_l, 10, 20)

        kp1_l, ds1_l = self.orb.detectAndCompute(img1_l, None)
        kp2_l, ds2_l = self.orb.detectAndCompute(img2_l, None)

        disparity = np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16)

        self.disparities.append(disparity)
        #imgtest = cv2.hconcat([disparity*4, img2_l.astype(np.float32), img2_r.astype(np.float32)])
        #plt.imsave("/home/meltem/thesis_orbslam/experiments/stereo_VO/disparities/img{i}.png".format(i=i), imgtest)

        #kp1_r, ds1_r = self.get_tiled_keypoints(img1_r, 10, 20) #10, 20
        #kp2_r, ds2_r = self.get_tiled_keypoints(img2_r, 10, 20)
        # Track the keypoints
        #tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        tp1_l, tp2_l = self.track_keypointsbymatch(img1_l, img2_l, kp1_l, kp2_l, ds1_l, ds2_l)

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)

        return transformation_matrix, self.matches, self.inliers

def main():
    dataset = "kitti"
    if dataset == "kitti":
        data_dir = "/media/meltem/moo/kitti/data_odometry_color/dataset/sequences/02"
        vo = VisualOdometry(data_dir, dataset, "", "")
    elif dataset == "own":   
        #data_dir = 'den_boer_mc0038_20220613_095019_sunny.bag'
        data_dir = "/media/meltem/moo/Own/den_boer_mc0038_20220613_095019_sunny.bag"
        calib_paths = ["/home/meltem/thesis_orbslam/calib_files_own/left_den_boer.yaml", "/home/meltem/thesis_orbslam/calib_files_own/right_den_boer.yaml"]
        poses_path = "/media/meltem/moo/Own/GT/den_boer/mc0038_20220613_095019_sunny.txt"
        vo = VisualOdometry(data_dir, dataset, poses_path, calib_paths)
    elif dataset == "flourish":
        data_dir = "/media/meltem/moo/Flourish/DatasetA.bag"
        vo = VisualOdometry(data_dir, dataset, "", "")


    
    #play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip
    gt_path = []
    
    estimated_path = []
    matches_list = []
    inliers_list = []

    #gt_pose = vo.gt_poses
    #gt_path = [(pose[0, 3], pose[1, 3]) for pose in gt_pose] 
    #for i in tqdm(list(range(len(vo.images_l))), unit="images"):
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i < 1:
            cur_pose = gt_pose
        else:
            transf, matches, inliers = vo.get_pose(i)
            cur_pose = np.matmul(cur_pose, transf)
            matches_list.append(matches)
            inliers_list.append(inliers)

        #gt_path.append((gt_pose[0, 3], gt_pose[1, 3]))
        #estimated_path.append((cur_pose[0, 3], cur_pose[1, 3]))
        #kitti:
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        i+=1
        #if i == 4745:#3:
        #    break
    with open('/home/meltem/thesis_orbslam/experiments/pathdata.csv', 'w') as f:
        write = csv.writer(f)
        for i in range(len(estimated_path)):
            write.writerow([gt_path[i][0], gt_path[i][1], estimated_path[i][0], estimated_path[i][1]])
            #write.writerow([estimated_path[i][0], estimated_path[i][1]])
    with open("/home/meltem/thesis_orbslam/experiments/stereo_VO/matches_inliers.txt", "w") as f:
        write = csv.writer(f)
        for i in range(len(matches_list)):
            write.writerow([matches_list[i], inliers_list[i]])

    plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
                        file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
