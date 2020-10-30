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
        self.bgr_path="rgb"
        self.depth_path="depth"

        #读取图片
        self.read_image_depth=None


    def see_images(self):
        bgr_paths=glob.glob(self.bgr_path+"/*.png")
        for bgr_image in bgr_paths:
            print("当前看的图片是:",bgr_image)
            #获取图片信息
            image_number=bgr_image.split("/")[-1]
            image_number=int(image_number[:-4])
            depth_path=self.depth_path+"/{}.png".format(image_number)

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
        指定图片读取,31,22,3,55,30
        :param image_number:
        :return:
        """
        image=cv.imread(self.bgr_path+"/{}.png".format(image_number))
        depth=cv.imread(self.depth_path+"/{}.png".format(image_number),cv.IMREAD_UNCHANGED)
        self.read_image_depth=depth.copy()
        return image,depth


    def read_depth_callback(self,event,x,y,flags,param):

        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.read_image_depth[y,x]
                print("HSV is :",hsv)
            except:
                return

def see_images():
    tradition_Way=Tradition_Way()
    tradition_Way.see_images()



def get_seg():
    tradition_Way=Tradition_Way()
    image,depth=tradition_Way.get_images(22)
    #查看图片
    color_depth=depth.copy()
    cv.normalize(color_depth,color_depth,255,0,cv.NORM_MINMAX)
    color_depth=color_depth.astype(np.uint8)
    color_map=cv.applyColorMap(color_depth,cv.COLORMAP_JET)
    cv.imshow("color_map",color_map)
    cv.imshow("image",image)
    cv.imshow("depth",depth)



    #进行图片分割
    cv.namedWindow("Read_Depth",cv.WINDOW_NORMAL)
    cv.imshow("Read_Depth",image)
    cv.setMouseCallback("Read_Depth",tradition_Way.read_depth_callback)


    ROI=cv.inRange(depth,0,680)
    cv.imshow("ROI",ROI)


    cv.waitKey(0)








if __name__ == '__main__':
    get_seg()
    # see_images()

