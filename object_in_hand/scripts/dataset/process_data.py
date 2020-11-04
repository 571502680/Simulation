"""
这里面用于查看场景数据和进行数据集清洗
"""
import os
import sys
import cv2 as cv
import numpy as np
import glob
import shutil


class Read_Scenes:
    def __init__(self,target_dataset_path=None,scenes_path=None):
        if target_dataset_path is None:
            self.target_dataset_path="./offical_dataset"
        else:
            self.target_dataset_path=target_dataset_path

        self.scenes_path=scenes_path

    def read_data(self):
        images_path=glob.glob(self.target_dataset_path+"/*_color.png")
        for image_path in images_path:
            depth_path=image_path[:-9]+"depth.png"


            image=cv.imread(image_path)
            depth=cv.imread(depth_path,cv.IMREAD_UNCHANGED)
            color_depth=depth.copy()
            cv.normalize(color_depth,color_depth,255,0,cv.NORM_MINMAX)
            color_depth=color_depth.astype(np.uint8)
            color_map=cv.applyColorMap(color_depth,cv.COLORMAP_JET)
            cv.imshow("color_map",color_map)
            cv.imshow("image",image)
            cv.waitKey(0)

    def get_scene_number(self,data):
        number=data.split("/")[-1]
        number=number.split("_")[0]
        return number

    ################################用于生成识别的要求的数据集###################################
    def generate_detect_dataset(self):
        """
        这里面用于生成一个文件夹,里面是自动标注软件要求的数据格式
        采用CenterNet直接进行训练,得到结果即可
        :return:
        """
        if self.scenes_path is None:
            print("[Error] self.scenes_path is None")

        number=0
        scenes_name=os.listdir(self.scenes_path)
        scenes_name=sorted(scenes_name,key=lambda data:int(data.split("-")[1]))
        for scene_path in scenes_name:
            images_path=glob.glob(self.scenes_path+"/"+scene_path+"/realsense/*_color.png")
            images_path=sorted(images_path,key=self.get_scene_number)#按照顺序进行整理
            for image_path in images_path:
                image=cv.imread(image_path)
                image_number=self.get_scene_number(image_path)
                depth_path=self.scenes_path+"/"+scene_path+"/realsense/{}_depth.png".format(image_number)
                depth=cv.imread(depth_path,cv.IMREAD_UNCHANGED)
                cv.imwrite(self.target_dataset_path+"/{}_depth.png".format(number),depth)
                cv.imwrite(self.target_dataset_path+"/{}_color.png".format(number),image)
                number=number+1

    def change_name(self):
        """
        为了让整体名称统一,因此进行所有文件的名称重命名
        :return:
        """
        number=0
        images_path=glob.glob(self.target_dataset_path+"/*_color.png")
        images_path=sorted(images_path,key=self.get_scene_number)#按照顺序进行整理

        for image_path in images_path:
            depth_path=image_path[:-9]+"depth.png"
            shutil.copy(image_path,self.target_dataset_path+"/{}_color.png".format(number))
            shutil.copy(depth_path,self.target_dataset_path+"/{}_depth.png".format(number))
            number=number+1

    ################################用于生成autolabel要求的数据集###################################
    def generate_autolabel_dataset(self):
        """
        这里面用于生成一个文件夹,里面是自动标注软件要求的数据格式
        采用CenterNet直接进行训练,得到结果即可
        :return:
        """
        if self.scenes_path is None:
            print("[Error] self.scenes_path is None")

        number=0
        scenes_name=os.listdir(self.scenes_path)
        scenes_name=sorted(scenes_name,key=lambda data:int(data.split("-")[1]))
        for scene_path in scenes_name:
            images_path=glob.glob(self.scenes_path+"/"+scene_path+"/realsense/*_color.png")
            images_path=sorted(images_path,key=self.get_scene_number)#按照顺序进行整理
            for image_path in images_path:
                image=cv.imread(image_path)
                cv.imwrite(self.target_dataset_path+"/{}.jpg".format(number),image)
                number=number+1




def read_images():
    read_Scenes=Read_Scenes()
    read_Scenes.read_data()

def chang_name():
    read_Scenes=Read_Scenes(target_dataset_path="all_detect_data",scenes_path="scenes")
    read_Scenes.generate_detect_dataset()
    # read_Scenes.change_name()

def generate_dataset():
    """
    用于生成自动标注软件要求的格式
    :return:
    """
    read_Scenes=Read_Scenes(target_dataset_path="autolabel_dataset",scenes_path="scenes")
    read_Scenes.generate_autolabel_dataset()




if __name__ == '__main__':
    # read_images()
    chang_name()
    # generate_dataset()


