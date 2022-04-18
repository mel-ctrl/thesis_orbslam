from scipy.spatial.transform import Rotation
import numpy as np
t_odom_c2 = np.array([0.667178, -0.4, 0])
r_odom_c2 = Rotation.from_euler('XYZ', [-0.016, 0.254, 0.0])


t_c1_odom = np.array((-0.667178, 0.28, 0.0))
r_c1_odom = (Rotation.from_euler('XYZ', [0.0, -0.23, 0.0])).inv()

t_c1_c2 = t_c1_odom+t_odom_c2
R_c1_c2 = Rotation.__mul__(r_c1_odom,r_odom_c2)
print(t_c1_c2)
print(R_c1_c2.as_matrix())