#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面进行DenseFusion数据集的生成任务,基于更改颜色的models和未更改颜色的models,生成DenseFusion要求的数据
Log: 2020.9.24:
    基于Read_Data进行数据读取,然后进行数据集的生成.HSV_node进行了更新,从而数据集的整合更好了
"""
import os
import sys
import rospy
import png
import cv2 as cv
import numpy as np
import open3d as o3d
import scipy.io as scio
#ROS中的通信协议
from cv_bridge import CvBridge
import Read_Data

python_path=os.path.dirname(__file__)
class Make_Data:
    def __init__(self,HSV_mode,init_node=False):
        """
        这个类用于制作数据集,HSV_mode情况下则生成语义分割图,否则生成RGB图
        :param HSV_mode:
        """
        self.HSV_mode=HSV_mode
        if init_node:
            rospy.init_node("Make_Data")

        #图像信息
        self.bridge=CvBridge()
        self.Camera_Info=np.array([[1120.1199067175087, 0.0, 640.5], [0.0, 1120.1199067175087], [360.5, 0.0, 0.0, 1.0]])
        #回调函数看值才使用
        self.depth_image=None
        self.bgr_image=None
        self.hsv_image=None

        #配置信息
        self.python_path=os.path.dirname(__file__)
        self.objects_names=self.get_objects_names()#用于存储物体名称
        self.all_index=self.get_all_index()#用于存储物体对应的bgr和hsv
        self.h_range=5

        #生成数据集路径
        try:
            self.materials_path=rospy.get_param('~materials_dir','/root/ocrtoc_materials')#默认是root路径下
        except:
            print("[Warning] ROS is not Start,the materials_path is default /root/ocrtoc_materials")
            self.materials_path='/root/ocrtoc_materials'
        self.dataset_pth=self.python_path+"/ALi_Dataset/data"

        #Pose初始化信息
        self.Trans_world2Camera=np.array([[-0.6184985,-0.14367474,-0.77253944,0.8138817 ],
                                          [ 0.12796457,-0.98843452,0.08137731,0.01304234],
                                          [-0.77529651,-0.04852593 ,0.62973054  ,0.07146605],
                                          [ 0.   ,       0.  ,        0.   ,       1.        ]])

    #############################生成需要的信息####################################
    def get_objects_names(self):
        """
        这里面进行此目录中classes.txt的文件更新
        :return:
        """
        file=open(self.python_path+"/classes.txt",'r')
        class_names=file.readlines()
        objects_name=[]
        for class_name in class_names:
            class_name=class_name[:-1]#去掉\n
            objects_name.append(class_name)
        return objects_name

    def get_all_index(self):
        """
        这里面进行all_index.txt的解析,获取hsv,bgr信息
        @return: self.all_index,是一个dict,每个object直接用真实名称命名,每个object内部包含'bgr'和'hsv'
        """
        temp_all_index={}
        all_index_file=open(self.python_path+"/all_index.txt","r")
        for line in all_index_file:
            object_name=line.split(':')[0]
            numbers=line.split(':')[1]
            numbers=numbers.split(',')
            numbers=list(map(int,numbers[:-2]))
            temp_all_index[object_name]={'bgr':[numbers[0],numbers[1],numbers[2]],'hsv':[numbers[3],numbers[4],numbers[5]]}
        return temp_all_index

    def see_label_image(self,scene_id="1-1"):
        """
        针对color_image生成对应的语义分割图,确定语义分割图是否正确
        :return:
        """
        #1:获取RGB图像
        read_Data=Read_Data.Read_Data()
        while not rospy.is_shutdown():
            bgr_image,depth_image=read_Data.get_images()
            if bgr_image is not None:

                #2:获取世界信息,对世界信息进行处理
                #2.1:寻找场景中所有的gazebo_name2true_name,gazebo_name_list等信息,从而进行图像分割
                mask=np.zeros(depth_image.shape,dtype=np.uint8)
                world_info_list,gazebo_name2true_name,gazebo_name_list=read_Data.get_world_info(scene_id=scene_id)

                #2.2:根据所有的true_name,采用hsv分割得到目标的Mask,一张全0的Mask进行相加即可
                hsv_image=cv.cvtColor(bgr_image,cv.COLOR_BGR2HSV)
                for gazebo_name in gazebo_name_list:
                    true_name=gazebo_name2true_name[gazebo_name]
                    hsv=self.all_index[true_name]['hsv']
                    low=np.array([max(0,hsv[0]-5),max(0,hsv[1]-30),0])
                    high=np.array([min(255,hsv[0]+5),min(255,hsv[1]+30),255])

                    temp_mask=cv.inRange(hsv_image,low,high)
                    mask_value=self.objects_names.index(true_name)+1
                    temp_mask=temp_mask/255*mask_value
                    temp_mask=temp_mask.astype(np.uint8)
                    mask=mask+temp_mask

                #2.3:进行Mask的展示
                mask=mask.astype(np.uint8)
                color_map=mask
                cv.normalize(color_map,color_map,255,0,cv.NORM_MINMAX)
                color_map=color_map.astype(np.uint8)
                color_map=cv.applyColorMap(color_map,cv.COLORMAP_JET)
                cv.imshow("color_map",color_map)
                cv.waitKey(0)
                break

    def read_image_hsv(self,event,x,y,flags,param):
        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.hsv_image[y,x]
                print("HSV is :",hsv)
            except:
                print("[Warning] Can not get HSV value")

    def read_image_bgr(self,event,x,y,flags,param):
        if event==cv.EVENT_MOUSEMOVE:
            try:
                bgr=self.bgr_image[y,x]
                print("BGR is :",bgr)
            except:
                print("[Warning] Can not get BGR value")

    def make_data(self,scene_id="1-1",debug=False):
        """
        这里面生成目标所需要的图片和Pose信息
        :param scene_id: 场景id数目
        :param debug:是否展示生成数据集的结果
        :return:
        """
        #1:调用需要使用的类
        if self.HSV_mode:
            read_Data=Read_Data.Read_Data(init_node=True,simulator='gazebo')#HSV的轮廓还是需要Gazebo输出
        else:
            read_Data=Read_Data.Read_Data(init_node=True,simulator='sapien')#RGB必须由sapien输出
        print("********Begin Make the Scene_id :{}********".format(scene_id))
        while not rospy.is_shutdown():
            #2:获取图像,世界信息等
            bgr_image,depth_image=read_Data.get_images()
            if bgr_image is not None and depth_image is not None:
                mask=np.zeros(depth_image.shape,dtype=np.uint8)
                world_info_list,gazebo_name2true_name,gazebo_name_list=read_Data.get_world_info(scene_id=scene_id)

                cls_indexes=[]
                poses=[]
                factor_depth=[10000]
                intrinsic_matrix=self.Camera_Info

                if self.HSV_mode:
                    #3:HSV模式中保存label_image,深度图,meta内容
                    #3.1:解析世界中的pose,name等信息
                    for each_object in world_info_list:
                        true_name=each_object['true_name']
                        class_id=self.objects_names.index(true_name)+1
                        model_pose=each_object['model_pose']

                        #继续获取他的对应4D姿态
                        Trans_raw2world=read_Data.get_matrix_from_modelpose(model_pose)
                        Trans_all=self.Trans_world2Camera.dot(Trans_raw2world)

                        cls_indexes.append(class_id)
                        poses.append(Trans_all[0:3,:])

                    #3.2:生成meta矩阵
                    poses=np.array(poses)
                    poses=np.transpose(poses,(1,2,0))
                    save_dict={}
                    save_dict['cls_indexes']=cls_indexes
                    save_dict['factor_depth']=factor_depth
                    save_dict['intrinsic_matrix']=intrinsic_matrix
                    save_dict['poses']=poses
                    scio.savemat(self.dataset_pth+"/{}-meta".format(scene_id),save_dict)
                    print("Already Make the Scene_id :{} meta".format(scene_id))

                    #3.3:保存label_image,depth_image
                    hsv_image=cv.cvtColor(bgr_image,cv.COLOR_BGR2HSV)
                    true_name_list=[]
                    for gazebo_name in gazebo_name_list:
                        true_name=gazebo_name2true_name[gazebo_name]
                        true_name_list.append(true_name)
                        hsv=self.all_index[true_name]['hsv']
                        low=np.array([max(0,hsv[0]-5),max(0,hsv[1]-30),0])
                        high=np.array([min(255,hsv[0]+5),min(255,hsv[1]+30),255])

                        temp_mask=cv.inRange(hsv_image,low,high)
                        mask_value=self.objects_names.index(true_name)+1
                        temp_mask=temp_mask/255*mask_value
                        temp_mask=temp_mask.astype(np.uint8)
                        mask=mask+temp_mask

                    if debug:
                        self.bgr_image=bgr_image
                        self.hsv_image=hsv_image
                        color_map=depth_image.copy()
                        cv.normalize(color_map,color_map,255,0,cv.NORM_MINMAX)
                        color_map=color_map.astype(np.uint8)
                        color_map=cv.applyColorMap(color_map,cv.COLORMAP_JET)
                        cv.namedWindow("color_map",cv.WINDOW_NORMAL)
                        cv.imshow("color_map",color_map)
                        cv.namedWindow("hsv_image",cv.WINDOW_NORMAL)
                        cv.imshow("hsv_image",hsv_image)
                        cv.setMouseCallback("hsv_image",self.read_image_hsv)#如果需要查看参数需要更新self.depth
                        cv.imshow("depth_image",depth_image)
                        cv.imshow("bgr_image",bgr_image)
                        cv.setMouseCallback("bgr_image",self.read_image_bgr)#如果需要查看参数需要更新self.depth
                        cv.imshow("mask",mask)
                        print("true_name_list: {}".format(true_name_list))
                        print("each hsv index:")
                        for true_name in true_name_list:
                            hsv=self.all_index[true_name]['hsv']
                            low=np.array([max(0,hsv[0]-5),max(0,hsv[1]-30),0])
                            high=np.array([min(255,hsv[0]+5),min(255,hsv[1]+30),255])
                            print("object:{},low:{},high:{}".format(true_name,low,high))
                        cv.waitKey(0)
                        sys.exit()#不进行图片的保存

                    #保存label图
                    print("Already Make the Scene_id :{} label_image and depth_image".format(scene_id))
                    cv.imwrite(self.dataset_pth+"/{}-label.png".format(scene_id),mask)

                    #保存深度图
                    depth_path=self.dataset_pth+"/{}-depth.png".format(scene_id)
                    with open(depth_path,'wb') as f:
                        writer=png.Writer(width=depth_image.shape[1],height=depth_image.shape[0],bitdepth=16)
                        zgray2list=depth_image.tolist()
                        writer.write(f,zgray2list)
                    break

                else:
                    #4:如果不是HSV mode,则只进行图片保存
                    if debug:
                        cv.imshow("bgr_image",bgr_image)
                        cv.waitKey(0)
                        sys.exit()#不进行结果的保存

                    print("Already Make the Scene_id :{} color_image".format(scene_id))
                    cv.imwrite(self.dataset_pth+"/{}-color.png".format(scene_id),bgr_image)
                    break

        print("********Already Make the Scene_id :{} data********".format(scene_id))

    #############################生成需要的信息####################################
    def check_gazebo_sapien_view_pose(self):
        """
        这里面确定gazebo和sapien的label图片是对应的
        :return:
        """
        sapien_bgr_image=cv.imread("temp_data/1-1-color.png")
        gazebo_label_image=cv.imread("temp_data/1-1-label.png")

        #label图片变换成3通道的
        cv_image=gazebo_label_image.copy()
        cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
        cv_image=cv_image.astype(np.uint8)
        color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)

        #两个图片融合起来
        merge_image=cv.addWeighted(sapien_bgr_image,0.5,color_map,1,0)
        cv.imshow("merge_image",merge_image)
        cv.waitKey(0)

    def check_object_pose(self):
        """
        读取物体的meta,,获取对应的xyz,另外对每个物体读取Pose,或取对应的world pose,看看最终是不是在一起的
        :return:
        """
        meta=scio.loadmat(self.python_path+"/temp_data/1-1-meta.mat")
        poses=meta['poses']
        print(poses)

def make_data():
    if len(sys.argv)!=3:
        print("[Error] Please input the scenid HSV_MODE to the make_data()")
        sys.exit()

    #配置对应场景
    scene_id=sys.argv[1]
    if sys.argv[2]=='True':
        make_Data=Make_Data(HSV_mode=True)
    elif sys.argv[2]=='False':
        make_Data=Make_Data(HSV_mode=False)#一定要记得改models
    else:
        print("[Error] The second input is Neither False or True,Please check")
        sys.exit()

    #制作数据集
    make_Data.make_data(scene_id=scene_id,debug=False)
    sys.exit()

def check_makecorrect(scene_id):
    """
    这里面主要确定pose是否正确,通过meta读取物体的pose(点云),然后通过仿真器获取物体的pose(axis),然后同时到open3d中显示
    :param scene_id:
    :return:
    """
    make_Data=Make_Data(HSV_mode=True)
    read_YCB=Read_Data.Read_YCB(get_object_points=True)
    read_Data=Read_Data.Read_Data(simulator='sapien')

    meta = scio.loadmat(make_Data.dataset_pth+'/{}-meta.mat'.format(scene_id))
    poses=meta['poses']
    cls_indexes=meta['cls_indexes']

    #从meta中获取物体pose,进行点云显示
    show_points=[]
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    show_points.append(axis_point)
    for i,class_id in enumerate(cls_indexes[0]):
        true_name=make_Data.objects_names[class_id-1]
        temp=np.zeros((4,4))
        pose=poses[:,:,i]
        temp[0:3,:]=pose
        temp[3,:]=[0,0,0,1]
        points=read_YCB.objects_points[true_name]
        target_o3d=o3d.geometry.PointCloud()
        target_o3d.points=o3d.utility.Vector3dVector(points)
        target_o3d.transform(temp)
        show_points.append(target_o3d)

    #从世界坐标系中获取物体pose,进行显示
    world_info_list,gazebo_name2true_name,gazebo_name_list=read_Data.get_world_info(scene_id=scene_id)
    for object_info in world_info_list:
        model_pose=object_info['model_pose']
        model_pose_matrix=read_Data.get_matrix_from_modelpose(model_pose)
        axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axis_point.transform(model_pose_matrix)
        axis_point.transform(read_Data.Trans_world2camera)
        show_points.append(axis_point)

    o3d.visualization.draw_geometries(show_points)


if __name__ == '__main__':
    # check_data()
    # check_makecorrect(scene_id='1-1')
    make_data()
