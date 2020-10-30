#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面快速测试一下,看看传统方法是否可行
全部基于Python3进行实现
"""
import glob
import cv2 as cv
import numpy as np





def see_images():
    images_path=glob.glob("data-10.20/*_color.jpg")
    for image_path in images_path:

        image=cv.imread(image_path)
        cv.imshow("image",image)
        cv.waitKey(0)


def see_images_1():
    image=cv.imread("temp_data/image.png")
    depth_image=cv.imread("temp_data/depth.png",cv.IMREAD_UNCHANGED)
    cv_image=depth_image.copy()

    cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
    cv_image=cv_image.astype(np.uint8)
    color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
    # cv.namedWindow("color_map",cv.WINDOW_NORMAL)
    cv.imshow("color_map",color_map)

    merge_image=cv.addWeighted(image,1.5,color_map,0.5,0)

    cv.imshow("image",image)
    cv.imshow("merge_image",merge_image)
    cv.waitKey(0)



if __name__ == '__main__':
    see_images_1()


