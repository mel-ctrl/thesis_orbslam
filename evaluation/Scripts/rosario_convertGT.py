import numpy as np
import argparse
import yaml
from scipy.spatial.transform import Rotation as R
def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

def new_points(quaternion, point):
    quaternion_inv = np.multiply(quaternion, np.array([1,-1,-1,-1]))
    point2 = np.multiply(quaternion,point,quaternion_inv)
    return point2

def listToString(s):
	# initialize an empty string
	str1 = ' '.join(map(str,s))
	
	# return string
	return (str1)

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.''')
    parser.add_argument('gt_original', help='original gt file (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('calib_file', help='yaml file with calibration params')
    parser.add_argument('gt_new', help='filename where to write new GT')
    parser.add_argument('inertial', help='is the method with intertial on, True or False?')
    #args = parser.parse_args()
    args = parser.parse_args(['/home/meltem/Downloads/sequence06_gt.txt', '/home/meltem/Downloads/calibration06.yaml', '/home/meltem/ORB_SLAM3/evaluation/Ground_truth/Rosario/sequence06_new_gt_left_cam_frame.txt','False'])


    lines_list = []

    with open(args.gt_original) as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            splitted = lines[i].split()
            lines_list.append(splitted)

    with open(args.calib_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        imu_p_baselink = np.array(params['imu']['position_imu_baselink'])
        imu_q_baselink = np.array(params['imu']['rotation_imu_baselink'])
        cam_left_p_imu = np.array(params['cam0']['T_cam_imu'])[:-1,3]
        cam_left_q_imu = R.from_matrix((np.array(params['cam0']['T_cam_imu'])[:-1,:-1])).as_quat()
    
    with open(args.gt_new, 'w') as f:
        for line in lines_list:
            q_point = np.array(line[-4:], dtype='float')
            p_point = np.array(line[1:4], dtype='float')
            new_time = format(float(line[0])*10**9, '.6f')
            if args.inertial == 'True':
                new_q_point = new_points(imu_q_baselink, q_point)
                new_p_point = p_point + imu_p_baselink
                new_line = str(new_time) +' '+ listToString(new_p_point.tolist()) +' '+ listToString(new_q_point.tolist()) + '\n'
                #new_line =  str(new_time) +' '+ listToString(new_p_point.tolist()) +' '+ listToString(q_point.tolist()) + '\n'
                f.write(new_line)    
                

            else:
                new_q = quaternion_multiply(imu_q_baselink, cam_left_q_imu)
                new_q_point = new_points(new_q, q_point)
                new_p_point = p_point + imu_p_baselink + cam_left_p_imu
                new_line = str(new_time) +' '+ listToString(new_p_point.tolist()) +' '+ listToString(new_q_point.tolist()) + '\n'
                #new_line =  str(new_time) +' '+ listToString(new_p_point.tolist()) +' '+ listToString(q_point.tolist()) + '\n'
                f.write(new_line)