#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import rospy
import time
import numpy as np
import cv2 as cv
import math
#msgs
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
import png
class Record_Data:
    def __init__(self):
        rospy.init_node("Record_Data")
        self.bridge=CvBridge()
        self.realsense_depth_image=None
        self.realsense_bgr_image=None


    def realsense_bgr_image_callback(self,data):
        try:
            self.realsense_bgr_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print("[Error] image_process_callback occur error {}".format(e))
            return

    def realsense_depth_callback(self,data,see_image=False):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,'32FC1')
            self.realsense_depth_image=cv_image*1000
            self.realsense_depth_image[np.isnan(self.realsense_depth_image)]=0
        except CvBridgeError as e:
            print("[Error] realsense_depth_callback occur error {}".format(e))
            return

        if see_image:
            if self.realsense_depth_image is  None:
                print("[Warning] Can not get the depth_image")
                return
            cv_image=self.realsense_depth_image.copy()
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
            cv.namedWindow("realsense_color_map",cv.WINDOW_NORMAL)
            cv.imshow("realsense_color_map",color_map)
            cv.waitKey(3)


    def begin_get_realsense_images(self):
        rospy.Subscriber("/realsense/depth/image_raw",Image,self.realsense_depth_callback)
        rospy.Subscriber("/realsense/color/image_raw",Image,self.realsense_bgr_image_callback)

if __name__ == '__main__':
    record_Data=Record_Data()
    record_Data.begin_get_realsense_images()

    record_data_count=0

    while not rospy.is_shutdown():
        if record_Data.realsense_bgr_image is not None:
            #1:展示图片
            cv.imshow("bgr_image",record_Data.realsense_bgr_image)
            cv_image=record_Data.realsense_depth_image.copy()
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
            cv.namedWindow("realsense_color_map",cv.WINDOW_NORMAL)
            cv.imshow("realsense_color_map",color_map)

            #2:进行图片收集
            input_info=cv.waitKey(3)
            if input_info==115:
                print("Save Image")
                cv.imwrite("{}_color.jpg".format(record_data_count),record_Data.realsense_bgr_image)
                cv.imwrite("{}_depth.png".format(record_data_count),record_Data.realsense_depth_image)
                record_data_count=record_data_count+1

