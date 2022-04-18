import rosbag
import argparse
import utm
from sensor_msgs.msg import NavSatFix, NavSatStatus
from scipy.spatial.transform import Rotation

def get_position(latitude, longitude, altitude):

    global initialx
    global initialy
    global initialz
    (x, y, zoneNumber, zoneLetter) = utm.from_latlon(latitude,longitude)
    z = altitude 
    if (initialx == 0):
        initialx = x
        initialy = y
        initialz = z

    # compute Position w.r.t the starting position
    gps_x = (x-initialx)
    gps_y = (y-initialy)
    gps_z = (z-initialz)

    return gps_x, gps_y, gps_z

def RTKtoOdomRef(dataset, x, y, z):
    if dataset == 'A':
        x = x - 0.11
        y = y - 0.38
        z = z + 2.19
    elif dataset == 'B':
        x = x - 0.11
        y = y - 0.38
        z = z + 2.67
    else:
        raise ValueError("Wrong dataset name, either A or B")
    return x, y, z

def OdomtoLeftCamRef(x, y, z):
    x = x + 0.667178
    y = y - 0.28
    z = z
    q_x, q_y, q_z, q_w = EulertoQuaternion()

    return x, y, z, q_x, q_y, q_z, q_w
    
    
    
def EulertoQuaternion():
    roll = 0.0
    pitch = -0.23
    yaw = 0.0
    q = Rotation.from_euler('XYZ', [roll, pitch, yaw])
    q = q.as_quat()
    print(q)
    return q

if __name__ == '__main__':
    bag = rosbag.Bag('/media/meltem/T7/Meltem/Thesis/Datasets/Flourish/DatasetB.bag')
    initialx = 0
    initialy = 0
    initialz = 0

    f_out = open("/home/meltem/Downloads/DatasetB_GT.txt", 'w+')

    for topic, msg, t in bag.read_messages(topics=['/sensor/gps/neomp8/ublox_gps_mp8/fix']):
        timestamp = msg.header.stamp.secs*10**9
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude 
        x, y, z = get_position(latitude, longitude, altitude)
        x, y, z = RTKtoOdomRef('B', x, y, z)
        x, y, z, q_x, q_y, q_z, q_w = OdomtoLeftCamRef(x, y, z)
        #print(x, y, z, q_x, q_y, q_z, q_w)
        f_out.write(str(timestamp) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(q_x) + ' ' + str(q_y) + ' ' + str(q_z) + ' ' + str(q_w) + "\r\n")
    
    bag.close()
    f_out.close()

