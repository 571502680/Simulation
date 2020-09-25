#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面尝试这进行点云的处理,看看能不能直接获取object的点云
"""
#comman package
import os
import sys
import rospy
import time
import numpy as np
import cv2 as cv
import open3d as o3d
#msgs
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
#my function
import robot_control

class See_Point_Cloud:
    def __init__(self,init_node=False):
        if init_node:
            rospy.init_node("See_Point_Cloud")

        #分别开启Realsense和Kinect的深度数据
        self.bridge=CvBridge()
        self.realsense_depth_image=None
        self.realsense_bgr_image=None
        self.kinect_depth_image=None
        self.kinect_bgr_image=None
        
        #####初始化中不开启两个深度摄像头,有需求的时候再进行调用,否则速度很慢
        # callback_lambda=lambda x:self.realsense_depth_callback(x,see_image=False)
        # realsense_depth_sub=rospy.Subscriber("/realsense/depth/image_raw",Image,callback_lambda)
        # callback_lambda=lambda x:self.kinect_depth_callback(x,see_image=False)
        # kinect_depth_sub=rospy.Subscriber("/kinect/depth/image_raw",Image,callback_lambda)
        #####初始化中不开启两个深度摄像头,有需求的时候再进行调用,否则速度很慢




    ####################################读取图片函数##################################
    def realsense_depth_callback(self,data,see_image=False):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,'32FC1')
            self.realsense_depth_image=cv_image*10000
            self.realsense_depth_image[np.isnan(self.realsense_depth_image)]=0
        except CvBridgeError as e:
            print("[Error] realsense_depth_callback occur error {}".format(e))
            return
        
        if see_image:
            if self.realsense_depth_image is  None:
                print("[Warning] Can not get the depth_image")
                return
            cv_image=self.realsense_depth_image.copy()

            # ROI=cv.inRange(cv_image,0,1500)#对于太远的去掉,省得看的并不明显
            # cv_image=cv_image*ROI/255
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
            cv.namedWindow("realsense_color_map",cv.WINDOW_NORMAL)
            cv.imshow("realsense_color_map",color_map)
            cv.waitKey(3)

    def kinect_depth_callback(self,data,see_image=False):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,'32FC1')
            self.kinect_depth_image=cv_image*10000
            self.kinect_depth_image[np.isnan(self.kinect_depth_image)]=0
        except CvBridgeError as e:
            print("[Error] kinect_depth_callback occur error {}".format(e))
            return

        if see_image:
            if self.kinect_depth_image is  None:
                print("[Warning] Can not get the depth_image")
                return
            cv_image=self.kinect_depth_image.copy()
            # ROI=cv.inRange(cv_image,0,1500)#对于太远的去掉,省得看的并不明显
            # cv_image=cv_image*ROI/255
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
            cv.namedWindow("kinect_color_map",cv.WINDOW_NORMAL)
            cv.imshow("kinect_color_map",color_map)
            cv.waitKey(3)

    def get_depth_images(self):
        if self.realsense_depth_image is not None and self.kinect_depth_image is not None:
            return self.realsense_depth_image,self.kinect_depth_image
        else:
            if self.realsense_depth_image is None:
                print("[Warning] bgr_image is None")
            if self.kinect_depth_image is None:
                print("[Warning] depth_image is None")
            time.sleep(0.5)#避免输出太快
            return None,None

    def temp(self):
        robot=robot_control.Robot()
        callback_lambda1=lambda x:self.realsense_depth_callback(x,see_image=True)
        realsense_depth_sub=rospy.Subscriber("/realsense/depth/image_raw",Image,callback_lambda1)
        callback_lambda2=lambda x:self.kinect_depth_callback(x,see_image=True)
        kinect_depth_sub=rospy.Subscriber("/kinect/depth/image_raw",Image,callback_lambda2)

        while not rospy.is_shutdown():
            robot.getpose_home(3)
            robot.home(t=3)



    def get_object_from_kinect(self):
        """
        通过kinect获取桌面潜在物体
        :return:
        """

        callback_lambda=lambda x:self.kinect_depth_callback(x,see_image=False)
        kinect_depth_sub=rospy.Subscriber("/kinect/depth/image_raw",Image,callback_lambda)

def see_depth_image():
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.temp()




if __name__ == '__main__':
    see_depth_image()









