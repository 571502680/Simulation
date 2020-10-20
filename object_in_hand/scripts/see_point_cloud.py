#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面尝试这进行点云的处理,看看能不能直接获取object的点云
希望是通过Kinect获取点云,然后去掉桌子的平面,得到的最终结果就是桌面上的物体.不过坐标系变换一直有问题,因此这个先放下暂时不管
"""
#comman package
import os
import sys
import rospy
import time
import numpy as np
import cv2 as cv
import open3d as o3d
import tf.transformations as trans_tools
import math

#msgs
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
#my function
import robot_control
import quaternion
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
    def realsense_bgr_image_callback(self,data):
        try:
            self.realsense_bgr_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print("[Error] image_process_callback occur error {}".format(e))
            return

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

    def begin_get_realsense_depth(self,see_image=False):
        callback_lambda=lambda x:self.realsense_depth_callback(x,see_image=see_image)
        realsense_depth_sub=rospy.Subscriber("/realsense/depth/image_raw",Image,callback_lambda)

    ####################################获取物体函数##################################
    def begin_get_object_realsense(self,see_image=False):
        """
        开启识别线程
        :param see_image:
        :return:
        """
        callback_lambda1=lambda x:self.process_realsense_depth_image(x,see_image=see_image)
        realsense_depth_sub=rospy.Subscriber("/realsense/depth/image_raw",Image,callback_lambda1)
        bgr_image=rospy.Subscriber("/realsense/color/image_raw",Image,self.realsense_bgr_image_callback)

    def process_realsense_depth_image(self,data,see_image=False):
        """

        :param data:
        :param see_image:
        :return:
        """
        #1:仍然是展示深度图
        try:
            self.realsense_depth_image=self.bridge.imgmsg_to_cv2(data,'32FC1')

            # self.realsense_depth_image=cv_image*10000#realsense的深度图返回的直接是mm
            self.realsense_depth_image[np.isnan(self.realsense_depth_image)]=0
            self.realsense_depth_image=self.realsense_depth_image*1000
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

        #2:inrange,得到mask图片
        print("max:{}  min:{}".format(np.max(self.realsense_depth_image),np.min(self.realsense_depth_image)))
        ROI=cv.inRange(self.realsense_depth_image,lowerb=0,upperb=280)
        cv.imshow("ROI",ROI)


        #然后采用区域分割,获取对应的bbox,得到bbox之后选择窄的方向生成抓取点
        _,contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            Rect=cv.boundingRect(contour)
            x,y,w,h=Rect
            cv.rectangle(self.realsense_bgr_image,(x,y),(x+w,y+h),(0,255,0),3)

        cv.imshow("bgr_image",self.realsense_bgr_image)
        cv.waitKey(1)

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

    def generate_move_points(self):
        """
        生成需要移动的目标
        x:0~0.25
        y:-0.35~0.35
        :return:
        """
        pre_pose=[]#采用一个list进行pose的保存

        #需要生成的角度信息
        pose_Matrix=trans_tools.euler_matrix(0,0,math.pi/2)
        pose_Matrix[0:3,3]=np.array([0.25,0,0.3]).T
        move_Matrix=pose_Matrix.dot(quaternion.euler_matrix(0,math.pi,0))
        rot=trans_tools.quaternion_from_matrix(move_Matrix)

        #xy进行生成
        x_point=3
        y_point=10
        for i in range(x_point):
            for j in range(y_point):
                x=i*0.1
                y=(j-3)*0.1
                trans=np.array([x,y,0.2])
                pose=np.hstack([trans,rot])
                pre_pose.append(pose)

        return pre_pose

    def get_object_from_realsense(self):
        """
        通过realsense获取桌面潜在物体
        :return:
        """
        #直接平行运动
        pass

def see_depth_image():
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.temp()

def see_object_from_realsense():
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.begin_get_object_realsense(see_image=True)
    robot=robot_control.Robot()
    move_pose=see_Point_Cloud.generate_move_points()

    while not rospy.is_shutdown():
        robot.getpose_home(t=1)
        # time.sleep(1)
        #整个桌面平行进行运动,z值固定
        for pose in move_pose:
            # print("Target Pose is :{}".format(pose))
            print pose
            arrive=robot.motion_generation(pose[np.newaxis,:],vel=0.2)
            if not arrive:
                robot.getpose_home()
                print("Arrive Failed,the target pose is:{}".format(pose))


if __name__ == '__main__':
    see_object_from_realsense()









