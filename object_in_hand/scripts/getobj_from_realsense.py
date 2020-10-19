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

        #realsense的相机参数
        self.fx=812.0000610351562
        self.fy=812.0000610351562
        self.cx=320.0
        self.cy=240.0

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
    def begin_get_realsense_images(self):
        """
        开启信息获取
        :return:
        """
        rospy.Subscriber("/realsense/depth/image_raw",Image,self.realsense_depth_callback)
        rospy.Subscriber("/realsense/color/image_raw",Image,self.realsense_bgr_image_callback)

    def begin_get_object_realsense(self,see_image=False):
        """
        开启识别线程
        :param see_image:
        :return:
        """
        callback_lambda1=lambda x:self.depth_process_callback(x)
        realsense_depth_sub=rospy.Subscriber("/realsense/depth/image_raw",Image,callback_lambda1)
        rospy.Subscriber("/realsense/color/image_raw",Image,self.realsense_bgr_image_callback)

    def generate_kernel(self,x,y):
        return np.ones((x,y),dtype=np.uint8)

    def process_realsense_depth_image(self,data,process_image=False,see_image=False):
        """
        用于处理深度图像,得到对应的轮廓
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
        ROI=cv.inRange(self.realsense_depth_image,lowerb=0,upperb=495)
        # cv.imshow("raw_ROI",ROI)
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(3,3))
        ROI=cv.morphologyEx(ROI,cv.MORPH_CLOSE,kernel=self.generate_kernel(6,6))
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(10,10))
        if see_image:
            cv.imshow("ROI",ROI)

        #然后采用区域分割,获取对应的bbox,得到bbox之后选择窄的方向生成抓取点
        _,contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            Rect=cv.minAreaRect(contour)
            draw_box=cv.boxPoints(Rect)
            draw_box=np.int0(draw_box)
            cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,0,255),3)

        cv.imshow("bgr_image",self.realsense_bgr_image)
        cv.waitKey(1)

    def process_rotate_rect(self,rotate_rect):
        """
        用于处理旋转矩形,得到一个长短边正确的旋转矩形
        :param rotate_rect:
        :return:
        """
        center,wh,xita=rotate_rect
        width=wh[0]
        height=wh[1]
        if width>height:
            xita=xita+90
            correct_width=height
            correct_height=width
            return center,(correct_width,correct_height),xita
        else:
            return rotate_rect

    def get_mask(self,see_image=False,debug=False):
        """
        通过深度分割,得到物体的center,然后对物体的center进行抓取操作
        :param see_image:
        :param debug:
        :return:
        """
        ROI=cv.inRange(self.realsense_depth_image,lowerb=0,upperb=495)
        if debug:
            print("max:{}  min:{}".format(np.max(self.realsense_depth_image),np.min(self.realsense_depth_image)))
            cv.imshow("raw_ROI",ROI)
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(3,3))
        ROI=cv.morphologyEx(ROI,cv.MORPH_CLOSE,kernel=self.generate_kernel(6,6))
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(10,10))
        if see_image:
            cv.imshow("ROI",ROI)

        #然后采用区域分割,获取对应的bbox,得到bbox之后选择窄的方向生成抓取点
        _,contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv.contourArea(contour)<3000:
                continue


            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确
            if see_image:
                #绘制物体轮廓
                draw_box=cv.boxPoints(rotate_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,0,255),3)

                #绘制抓取中心点
                grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])
                draw_box=cv.boxPoints(grasp_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,255,0),3)
                if debug:
                    print("Find good rect:{}".format(rotate_rect))
                    print("contour area:{}".format(cv.contourArea(contour)))#最小的认为是5000

        if see_image:
            cv.imshow("bgr_image",self.realsense_bgr_image)
            cv.waitKey(0)

    def get_centers(self,see_image=False,debug=False):
        """
        通过深度分割,得到物体的center,然后对物体的center进行抓取操作
        :param see_image:
        :param debug:
        :return:
        """
        self.realsense_depth_image=None
        while True:
            if self.realsense_depth_image is not None:
                break
            else:
                time.sleep(0.5)
                print("[Warning] get_centers self.realsense_depth_image is None")
        centers=[]
        ROI=cv.inRange(self.realsense_depth_image,lowerb=0,upperb=495)
        if debug:
            print("max:{}  min:{}".format(np.max(self.realsense_depth_image),np.min(self.realsense_depth_image)))
            cv.imshow("raw_ROI",ROI)
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(3,3))
        ROI=cv.morphologyEx(ROI,cv.MORPH_CLOSE,kernel=self.generate_kernel(6,6))
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(10,10))
        if see_image:
            cv.imshow("ROI",ROI)

        #然后采用区域分割,获取对应的bbox,得到bbox之后选择窄的方向生成抓取点
        _,contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv.contourArea(contour)<3000:
                continue
            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确
            centers.append(rotate_rect[0])
            if see_image:
                #绘制物体轮廓
                draw_box=cv.boxPoints(rotate_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,0,255),3)
                cv.putText(self.realsense_bgr_image,"{}".format(len(centers)),(int(rotate_rect[0][0]),int(rotate_rect[0][1])),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

                #绘制抓取中心点
                # grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])
                # draw_box=cv.boxPoints(grasp_rect)
                # draw_box=np.int0(draw_box)
                # cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,255,0),3)
                if debug:
                    print("Find good rect:{}".format(rotate_rect))
                    print("contour area:{}".format(cv.contourArea(contour)))#最小的认为是5000

        if see_image:
            cv.imshow("Center BGR Image",self.realsense_bgr_image)
            cv.waitKey(1)

        return centers

    def get_xyz_from_point(self,center,range_area=2):
        """
        这里面通过给定点获取对应的xyz值
        :param center: 图像中心点
        :return:
        """
        u,v=center
        u=int(u)
        v=int(v)
        center_Z=[]
        #1:对center_Z进行排序,得到中值作为深度
        try:
            for x in range(-range_area,range_area+1):
                for y in range(-range_area,range_area+1):
                    center_Z.append(self.realsense_depth_image[v-y,u-x])#采用行列索引
            center_Z.sort()
            Z=center_Z[int(len(center_Z)/2)]
        except:
            try:
                Z=self.realsense_depth_image[v,u]
            except:
                Z=0

        #2:使用外参进行反解
        X=(u-self.cx)*Z/self.fx
        Y=(v-self.cy)*Z/self.fy
        return X,Y,Z

    def get_grasprect(self,grasp_number,see_image=False,debug=False):
        """
        这里面进行精确的抓取点获取,从而进行抓取执行
        :return:
        """
        best_grasp_rect=None#抓取矩形为None
        pre_rects=[]

        #1:等待图片更新
        self.realsense_depth_image=None
        while True:
            if self.realsense_depth_image is not None:
                break
            else:
                time.sleep(0.5)
                print("[Warning] get_grasprect self.realsense_depth_image is None")
        ROI=cv.inRange(self.realsense_depth_image,lowerb=0,upperb=295)#这里深度阈值需要改一下
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(3,3))
        ROI=cv.morphologyEx(ROI,cv.MORPH_CLOSE,kernel=self.generate_kernel(6,6))
        ROI=cv.morphologyEx(ROI,cv.MORPH_OPEN,kernel=self.generate_kernel(10,10))
        if debug:
            cv.imshow("grasprec_ROI",ROI)

        #2:采用区域分割,获取对应的bbox,得到bbox之后选择窄的方向生成抓取点
        _,contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv.contourArea(contour)<5000:
                continue
            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确
            if debug:
                #绘制物体轮廓
                draw_box=cv.boxPoints(rotate_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,0,255),3)

            #3:获取抓取矩形
            grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])
            pre_rects.append(grasp_rect)

        if len(pre_rects)>1:
            width=1000
            correct_rect=None
            for pre_rect in pre_rects:
                distance=abs(pre_rect[0][0]-320)+abs(pre_rect[0][1]-240)
                if distance<width:
                    correct_rect=pre_rect
                    width=distance
        elif len(pre_rects)==1:
            correct_rect=pre_rects[0]
        else:
            return None

        #3:绘制抓取矩形
        draw_box=cv.boxPoints(correct_rect)
        draw_box=np.int0(draw_box)
        cv.drawContours(self.realsense_bgr_image,[draw_box],0,(0,255,0),3)
        cv.putText(self.realsense_bgr_image,"{}".format(grasp_number),(30,30),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3)

        if debug:
            print("Find good rect:{}".format(correct_rect))
            print("contour area:{}".format(cv.contourArea(correct_rect)))#最小的认为是5000

        if see_image:
            cv.imshow("grasprect_BGR_image",self.realsense_bgr_image)
            cv.waitKey(0)

        return best_grasp_rect

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

    def generate_realsense_movepoints(self):
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
        move_Matrix=pose_Matrix.dot(quaternion.euler_matrix(0,math.pi/2,0))
        rot=trans_tools.quaternion_from_matrix(move_Matrix)

        #xy进行生成
        x_point=3
        y_point=10
        for i in range(x_point):
            for j in range(y_point):
                x=i*0.1+0.1
                y=(j-3)*0.1
                trans=np.array([x,y,0.5])
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
    see_Point_Cloud.begin_get_object_realsense(see_image=False)
    robot=robot_control.Robot()
    move_pose=see_Point_Cloud.generate_realsense_movepoints()
    robot.getpose_home(t=1)
    while not rospy.is_shutdown():
        robot.getpose_home(t=1)
        time.sleep(1)
        #整个桌面平行进行运动,z值固定
        for pose in move_pose:
            # print("Target Pose is :{}".format(pose))
            arrive=robot.motion_generation(pose[np.newaxis,:],vel=0.5)
            if not arrive:
                robot.getpose_home()
                print("Arrive Failed,the target pose is:{}".format(pose))

def get_correct_realsense_pose():
    """
    这里面调整角度,使realsense正对桌面,而不是存在一个角度.不过这里面需要直接改robot_control的逆解代码.这个之后需要进行跟进
    :return:
    """
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.begin_get_object_realsense(see_image=True)
    robot=robot_control.Robot()

    pose_Matrix=trans_tools.euler_matrix(0,0,math.pi/2)
    pose_Matrix[0:3,3]=np.array([0.25,0,0.3]).T
    move_Matrix=pose_Matrix.dot(quaternion.euler_matrix(0,math.pi/2,0))
    rot=trans_tools.quaternion_from_matrix(move_Matrix)
    trans=[0.1,-0.15,0.5]
    pose=np.hstack([trans,rot])
    robot.getpose_home(1)
    while not rospy.is_shutdown():
        # robot.getpose_home(t=1)
        robot.motion_generation(pose[np.newaxis,:],vel=0.2)
        time.sleep(20)
        # break

def get_grasp_rect():
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.begin_get_realsense_images()
    robot=robot_control.Robot()
    move_pose=see_Point_Cloud.generate_realsense_movepoints()#获取桌面运动点,进行桌面平动
    robot.getpose_home(t=1)
    while not rospy.is_shutdown():
        # robot.getpose_home(t=1)
        #1:运动到待抓取位置
        for pose in move_pose:
            # print("Target Pose is :{}".format(pose))
            arrive=robot.motion_generation(pose[np.newaxis,:],vel=0.5)

            if not arrive:
                robot.getpose_home()
                print("Arrive Failed,the target pose is:{}".format(pose))

            #2:解析深度图得到对应的Mask,然后获取图像中的中心点:
            if see_Point_Cloud.realsense_depth_image is not None:
                see_Point_Cloud.get_mask(see_image=True,debug=False)

def move_object_upper():
    """
    用于运动到检测到的物体上方
    :return:
    """
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.begin_get_realsense_images()
    robot=robot_control.Robot()
    move_pose=see_Point_Cloud.generate_realsense_movepoints()#获取桌面运动点,进行桌面平动
    robot.getpose_home(t=1)
    while not rospy.is_shutdown():
        robot.getpose_home(t=1)
        #1:运动到待抓取位置
        for pose in move_pose:
            # print("Target Pose is :{}".format(pose))
            arrive=robot.motion_generation(pose[np.newaxis,:],vel=0.5)
            time.sleep(0.5)

            if not arrive:
                robot.getpose_home()
                print("Arrive Failed,the target pose is:{}".format(pose))

            #2:解析深度图得到对应的Mask,然后获取图像中的中心点:
            centers=see_Point_Cloud.get_centers(see_image=True,debug=False)
            for count_i,center in enumerate(centers):
                x,y,z=see_Point_Cloud.get_xyz_from_point(center)
                #移动到物体上方
                temp_pose=pose.copy()
                temp_pose[0]=temp_pose[0]+x/1000#变换到m制度
                temp_pose[1]=temp_pose[1]-y/1000
                temp_pose[2]=temp_pose[2]-0.2#降低Z值,从而尽可能地只看到一个物体
                robot.motion_generation(temp_pose[np.newaxis,:],vel=0.5)
                #获取更精确的抓取目标
                see_Point_Cloud.get_grasprect(grasp_number=count_i,see_image=True)

            print("Robot will go to the next big Pose")

def get_grasp_pose():
    """
    运动到物体上方,再进行抓取位置精修,从而最终完成抓取任务
    :return:
    """
    see_Point_Cloud=See_Point_Cloud(init_node=True)
    see_Point_Cloud.begin_get_realsense_images()
    robot=robot_control.Robot()
    move_pose=see_Point_Cloud.generate_realsense_movepoints()#获取桌面运动点,进行桌面平动
    robot.getpose_home(t=1)
    while not rospy.is_shutdown():
        robot.getpose_home(t=1)
        time.sleep(1)
        #1:运动到待抓取位置
        for pose in move_pose:
            # print("Target Pose is :{}".format(pose))
            arrive=robot.motion_generation(pose[np.newaxis,:],vel=0.5)
            time.sleep(0.5)

            if not arrive:
                robot.getpose_home()
                print("Arrive Failed,the target pose is:{}".format(pose))

            #2:解析深度图得到对应的Mask,然后获取图像中的中心点:
            centers=see_Point_Cloud.get_centers(see_image=True,debug=False)
            for count_i,center in enumerate(centers):
                x,y,z=see_Point_Cloud.get_xyz_from_point(center)

                #3:运动到物体上方
                temp_pose=pose.copy()
                temp_pose[0]=temp_pose[0]+x/1000#变换到m制度
                temp_pose[1]=temp_pose[1]-y/1000
                temp_pose[2]=temp_pose[2]-0.2#降低Z值,从而尽可能地只看到一个物体
                robot.motion_generation(temp_pose[np.newaxis,:],vel=0.5)
                #获取更精确的抓取目标
                see_Point_Cloud.get_grasprect(grasp_number=count_i,see_image=True)

                #4:执行抓取任务
            print("Robot will go to the next big Pose")

if __name__ == '__main__':
    move_object_upper()









