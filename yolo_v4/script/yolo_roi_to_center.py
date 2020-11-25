#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
sys.path.insert(1, '/usr/local/lib/python3.6/dist-packages/cv2')
import cv2
from yolo_v4.msg import ROI
from yolo_v4.msg import ROI_array
from yolo_v4.msg import ROI_center_array
import rospy
from absl import app, flags, logging

default_z = rospy.get_param('~/yolo_roi_to_center/default_z')

def ROI_array_callback(roi_array):
    ROI_List = roi_array.ROI_list

    if len(ROI_List) > 0:
        print("=============")
        for i in range(len(ROI_List)):
            print("Object_{} : {} ( x_min:{}, y_main:{}, x_max:{}, y_max:{})".format(i+1, ROI_List[i].object_name, ROI_List[i].min_x, ROI_List[i].min_y, ROI_List[i].Max_x, ROI_List[i].Max_y))
        print("=============")
        object_name = ROI_List[0].object_name
        score = ROI_List[0].score
        x_center = int((ROI_List[0].min_x + ROI_List[0].Max_x)/2)
        y_center = int((ROI_List[0].min_y + ROI_List[0].Max_y)/2)
        roi_center_pub.publish(object_name, score, x_center, y_center, default_z)

if __name__ == '__main__':
    try:
        rospy.init_node('get_yolo_v4_ROI_array node', anonymous=True)
        rospy.Subscriber("/object/ROI_array", ROI_array, ROI_array_callback)
        roi_center_pub = rospy.Publisher("/object/Center_array", ROI_center_array, queue_size=100)
        rate = rospy.Rate(10) # 10hz
        rospy.spin()
    except SystemExit:
        pass

