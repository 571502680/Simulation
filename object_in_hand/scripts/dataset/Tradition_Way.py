#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面快速测试一下,看看传统方法是否可行
全部基于Python3进行实现
"""
import glob
import cv2 as cv
import numpy as np


class Tradition_Way:
    def __init__(self):
        self.images_path="MyFile"

        #读取图片
        self.read_image_depth=None

    ##############################读取图片函数##############################
    def see_images(self):
        bgr_paths=glob.glob(self.images_path+"/*_color.png")
        for bgr_image in bgr_paths:
            print("当前看的图片是:",bgr_image)
            #获取图片信息
            image_number=bgr_image.split("/")[-1]
            image_number=image_number.split("_")[0]
            depth_path=self.images_path+"/{}_depth.png".format(image_number)

            #获取深度和图片
            image=cv.imread(bgr_image)
            depth=cv.imread(depth_path,cv.IMREAD_UNCHANGED)#深度为mm制
            self.read_image_depth=depth.copy()
            ROI=cv.inRange(depth,0,1500)#对于太远的去掉,省得看的并不明显
            depth=depth*ROI/255
            cv_image=depth.copy()
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)

            #读取深度图参数,确定比较好的深度图
            cv.namedWindow("Read_Depth",cv.WINDOW_NORMAL)
            cv.imshow("Read_Depth",image)
            cv.setMouseCallback("Read_Depth",self.read_depth_callback)

            #最后进行图片展示
            cv.imshow("color_map",color_map)
            cv.imshow("image",image)
            cv.waitKey(0)

    def get_images(self,image_number):
        """
        :param image_number:
        :return:
        """
        image=cv.imread(self.images_path+"/{}_color.png".format(image_number))
        depth=cv.imread(self.images_path+"/{}_depth.png".format(image_number),cv.IMREAD_UNCHANGED)
        self.read_image_depth=depth.copy()
        return image,depth

    def read_depth_callback(self,event,x,y,flags,param):

        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.read_image_depth[y,x]
                print("HSV is :",hsv)
            except:
                return

    ##############################基于深度图进行图像切割##############################
    def get_roi(self,depth_range,depth_image,image,see_depth=False,debug=False):
        """
        基于depth_range获取深度图图像
        :param depth_range: cv.inRange的参数
        :param depth_image: 深度图
        :param image: BGR图,用于进行debug
        :param see_depth: 是否查看图片的深度值
        :param debug: 是否查看color_map用于调参
        :return:
        """

        #查看图片
        if debug:
            color_depth=depth_image.copy()
            cv.normalize(color_depth,color_depth,255,0,cv.NORM_MINMAX)
            color_depth=color_depth.astype(np.uint8)
            color_map=cv.applyColorMap(color_depth,cv.COLORMAP_JET)
            cv.imshow("color_map",color_map)

        #进行图片分割
        if see_depth:
            cv.namedWindow("Read_Depth",cv.WINDOW_NORMAL)
            cv.imshow("Read_Depth",image)
            cv.setMouseCallback("Read_Depth",self.read_depth_callback)

        ROI=cv.inRange(depth_image,0,depth_range)
        if debug:
            cv.imshow("ROI",ROI)
            cv.waitKey(0)
        return ROI


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

    def get_grasp_rect(self,ROI,image,debug=False,see_image=False):
        pre_rects=[]
        best_grasp_rect=None

        #1:直接基于深度图分割
        contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour)<1000:
                continue

            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确
            if debug:
                #绘制物体轮廓
                draw_box=cv.boxPoints(rotate_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(image,[draw_box],0,(0,0,255),3)

            #3:获取抓取矩形
            grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])
            pre_rects.append(grasp_rect)

            if len(pre_rects)<1:
                return None

            #2:进行抓取矩形框绘制
            draw_box=cv.boxPoints(grasp_rect)
            draw_box=np.int0(draw_box)
            cv.drawContours(image,[draw_box],0,(0,255,0),3)

        if see_image:
            cv.imshow("grasprect_BGR_image",image)
            cv.waitKey(0)

        return best_grasp_rect


def see_images():
    tradition_Way=Tradition_Way()
    tradition_Way.see_images()

def get_seg():
    """
    基于depth进行图像分割,得到分割图片
    :param depth: 深度图
    :param see_depth: 是否使用回调函数查看图片
    :param debug: 是否进行debug
    :return:
    """
    tradition_Way=Tradition_Way()
    for number in range(20):
        image,depth=tradition_Way.get_images(number)
        ROI=tradition_Way.get_roi(720,depth,image)
        tradition_Way.get_grasp_rect(ROI,image,debug=True,see_image=True)




if __name__ == '__main__':
    get_seg()
    # see_images()

