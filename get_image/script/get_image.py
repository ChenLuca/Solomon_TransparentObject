#!/usr/bin/env python3
import sys
#sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/cv2")
#sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
# from get_image.srv import *
import datetime 
import cv2
from cv_bridge import CvBridge, CvBridgeError

import time 
import argparse
import numpy as np
import os
parser = argparse.ArgumentParser()
parser.add_argument('--Object_Name', type=str, default='.', help='Class name of training object.')
FLAGS = parser.parse_args()

day = str(datetime.datetime.now()).split(" ")[0]
time = str(datetime.datetime.now()).split(" ")[1]
time = time.split(":")[0] + "_" + time.split(":")[1] + "_" + time.split(":")[2].split(".")[0]

current_time = day + "_" + time + "_"

Object_Name = FLAGS.Object_Name

Train_Data_Dir = os.path.dirname(os.path.realpath(__file__)) + '/Training_Data/' + \
    current_time + '_' + Object_Name + '/'

value = None

class Get_image():
    def __init__(self):
        rospy.init_node('get_image_from_Realsense_D435i', anonymous=True)
        
        self.bridge = CvBridge()
        self.image = np.zeros((0,0,3), np.uint8)
        self.take_picture_counter = 0

        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

        if not os.path.exists(Train_Data_Dir):
            os.makedirs(Train_Data_Dir)
        
        rospy.spin()

    def callback(self, image):
        global value
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", self.cv_image)
        self.get_image(self.cv_image)
        cv2.waitKey(1)

    def get_image(self, crop_image):
        if cv2.waitKey(33) & 0xFF == ord('s'):
            name = str(Train_Data_Dir + current_time + '_' + Object_Name + '_' + str(self.take_picture_counter+1) + ".jpg")
            cv2.imwrite(name,crop_image)
            print("[Save] ", name)
            self.take_picture_counter += 1
        else:
            pass

if __name__ == '__main__':
    listener = Get_image()
    cv2.destroyAllWindows()
