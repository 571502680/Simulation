#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面尝试这进行点云的处理,看看能不能直接获取object的点云
2020.10.18:
    希望通过realsense获取物体的矩形框,然后尝试进行抓取,看看是否能成功抓取物体.要完成抓取一个物体,然后运动到kinect摄像头面前的工作.
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
        rospy.Subscriber("/realsense/color/image_raw",Image,self.realsense_bgr_image_callback)

    def generate_kernel(self,x,y):
        return np.ones((x,y),dtype=np.uint8)

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
        ROI=cv.inRange(self.realsense_depth_image,lowerb=0,upperb=480)
        cv.imshow("raw_ROI",ROI)
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(3,3))
        ROI=cv.morphologyEx(ROI,cv.MORPH_CLOSE,kernel=self.generate_kernel(6,6))
        cv.imshow("ROI",ROI)

        #然后采用区域分割,获取对应的bbox,得到bbox之后选择窄的方向生成抓取点
        _,contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            Rect=cv.boundingRect(contour)
            x,y,w,h=Rect
            cv.rectangle(self.realsense_bgr_image,(x,y),(x+w,y+h),(0,255,0),3)

        cv.imshow("bgr_image",self.realsense_bgr_image)
        cv.waitKey(1)

    def see_depth_image(self):
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
    see_Point_Cloud.see_depth_image()

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
            arrive=robot.motion_generation(pose[np.newaxis,:],vel=0.5)
            if not arrive:
                robot.getpose_home()
                print("Arrive Failed,the target pose is:{}".format(pose))

def get_correct_realsense_pose():
    """
    这里面调整角度,使realsense正对桌面,而不是存在一个角度
    :return:
    """
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.begin_get_object_realsense(see_image=True)
    robot=robot_control.Robot()

    pose_Matrix=trans_tools.euler_matrix(0,0,math.pi/2)
    pose_Matrix[0:3,3]=np.array([0.25,0,0.3]).T
    move_Matrix=pose_Matrix.dot(quaternion.euler_matrix(0,math.pi/2,0))
    # move_Matrix=pose_Matrix.dot(quaternion.euler_matrix(0,math.pi,0))
    rot=trans_tools.quaternion_from_matrix(move_Matrix)
    trans=[0.1,-0.15,0.5]
    pose=np.hstack([trans,rot])
    robot.getpose_home(1)
    while not rospy.is_shutdown():
        # robot.getpose_home(t=1)
        robot.motion_generation(pose[np.newaxis,:],vel=0.3)
        time.sleep(20)
        # break








if __name__ == '__main__':
    get_correct_realsense_pose()









