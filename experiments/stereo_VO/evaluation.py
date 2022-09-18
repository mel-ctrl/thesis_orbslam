
import numpy as np
from lib.visualization import plotting
import os
def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """


    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = np.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    
    
    transGT = data.mean(1) - s*rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_alignedGT = s*rot * model + transGT
    model_aligned = rot * model + trans

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = np.sqrt(np.sum(np.multiply(alignment_errorGT,alignment_errorGT),0)).A[0]
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,transGT,trans_errorGT,trans,trans_error, s

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
            #T_rotate = np.empty((4, 4))
            #T_rotate[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
            #T_rotate[:3, 3] = [0, 0, 0]
            #T_rotate[3, :] = [0, 0, 0, 1]
            #T_final = np.matmul(T, T_rotate)
            poses.append(T)
    return poses

estimated_poses = _load_poses('/home/meltem/thesis_orbslam/CameraTrajectory.txt')
gt_poses = _load_poses('/media/meltem/moo/kitti/GT/02.txt')

rot,transGT,trans_errorGT,trans,trans_error, scale = align(estimated_poses,gt_poses)
estimated_poses_aligned = scale * rot * estimated_poses + transGT
gt_path = []
estimated_path = []

output = "/home/meltem/thesis_orbslam/experiments/stereo_VO/test"
for i, gt_pose in enumerate(gt_poses):
    gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
    if i == len(estimated_poses)-1:
        break
for cur_pose in estimated_poses_aligned:
    estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
                    file_out=os.path.basename(output) + ".html")
