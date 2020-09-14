#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面进行DenseFusion数据集的生成任务,基于更改颜色的models和未更改颜色的models,生成DenseFusion要求的数据
"""
import os
import sys
import rospy
import png
import cv2 as cv
import numpy as np
import tf
import tf.transformations as trans_tools
import open3d as o3d
import scipy.io as scio
import xml.etree.ElementTree as ET
#ROS中的通信协议
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
from gazebo_msgs.srv import GetModelState,SetModelState,GetModelStateRequest,GetWorldProperties,GetWorldPropertiesRequest,SpawnModel,SpawnModelRequest
from geometry_msgs.msg import Pose,Point,Quaternion

import Read_Data


class Make_Data:
    def __init__(self):
        """
        这个类用于制作数据集.先解决一个场景的制作,然后再说其他的问题
        """
        #图像信息
        self.bridge=CvBridge()
        self.Camera_Info=np.array([[1120.1199067175087, 0.0, 640.5], [0.0, 1120.1199067175087], [360.5, 0.0, 0.0, 1.0]])
        self.depth_image=None
        self.color_image=None
        self.hsv_image=None

        #配置信息
        self.objects_names=['a_cups', 'a_lego_duplo', 'a_toy_airplane', 'adjustable_wrench', 'b_cups', 'b_lego_duplo', 'b_toy_airplane', 'banana', 'bleach_cleanser', 'bowl', 'bowl_a', 'c_cups', 'c_lego_duplo', 'c_toy_airplane', 'cracker_box', 'cup_small', 'd_cups', 'd_lego_duplo', 'd_toy_airplane', 'e_cups', 'e_lego_duplo', 'e_toy_airplane', 'extra_large_clamp', 'f_cups', 'f_lego_duplo', 'flat_screwdriver', 'foam_brick', 'fork', 'g_cups', 'g_lego_duplo', 'gelatin_box', 'h_cups', 'hammer', 'i_cups', 'j_cups', 'jenga', 'knife', 'large_clamp', 'large_marker', 'master_chef_can', 'medium_clamp', 'mug', 'mustard_bottle', 'nine_hole_peg_test', 'pan_tefal', 'phillips_screwdriver', 'pitcher_base', 'plate', 'potted_meat_can', 'power_drill', 'prism', 'pudding_box', 'rubiks_cube', 'scissors', 'spoon', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can', 'wood_block']
        self.python_path=os.path.dirname(__file__)
        self.all_index={}

        #生成数据集路径
        self.dataset_pth=self.python_path+"/../ALi_dataset/Dataset"

    #############################生成需要的信息####################################
    def get_all_index(self):
        """
        这里面进行all_index.txt的解析,获取hsv,bgr信息
        @return: self.all_index,是一个dict,每个object直接用真实名称命名,每个object内部包含'bgr'和'hsv'
        """
        all_index_file=open(self.python_path+"/all_index.txt")
        for line in all_index_file:
            object_name=line.split(':')[0]
            numbers=line.split(':')[1]
            numbers=numbers.split(',')
            numbers=list(map(int,numbers[:-2]))
            self.all_index[object_name]={'bgr':[numbers[0],numbers[1],numbers[2]],'hsv':[numbers[3],numbers[4],numbers[5]]}
        return self.all_index

    def get_worldfile_path(self,scene_id=None):
        materials_path = rospy.get_param('~materials_dir','/root/ocrtoc_materials')
        scenes_dir = materials_path + '/scenes'
        # rospy.loginfo("scenes dir: " + scenes_dir)

        try:#可以开启trigger_and_score.py文件,指定场景名称,不开启默认使用1-1的场景
            task_name = rospy.get_param('~scene')
        except:
            if scene_id is None:
                print("[Warning] Not Use trigger_and_score.py,Default use scene:1-1")
                task_name="1-1"
            else:
                task_name=scene_id

        task_path = scenes_dir + "/" + task_name + "/input.world"
        return task_path

    def get_worldfile_sys_path(self):
        """
        这里面直接读取python的命令行代码返回场景id
        @return:
        """
        materials_path = rospy.get_param('~materials_dir','/root/ocrtoc_materials')
        scenes_dir = materials_path + '/scenes'
        task_name=sys.argv[1]
        task_path = scenes_dir + "/" + task_name + "/input.world"
        return task_path

    #############################进行不同.world文件导入############################
    def change_world(self):
        """
        这个函数没有用,一直有bug,导致插入不了新的world
        @return:
        """

        rospy.init_node("Temp")
        # task_path=self.get_worldfile_path()
        task_path=self.get_worldfile_path()


        # initial_pose = Pose()
        # initial_pose.position.x = 1
        # initial_pose.position.y = 1
        # initial_pose.position.z = 1

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        insert_new_world=rospy.ServiceProxy('/gazebo/spawn_sdf_model',SpawnModel)

        request=SpawnModelRequest()

        f=open(task_path,'r')
        request.model_name="toy1"
        request.model_xml=f.read()
        request.robot_namespace="robot_namespace"
        # request.initial_pose=Pose
        request.reference_frame="world"


        insert_new_world('ground_plane', open(task_path,'r').read(), "/foo", Pose(position= Point(0,0,2),orientation=Quaternion(0,0,0,0)),"world")

        # insert_new_world(request)
        # print("temp",temp)

    #############################进行图片处理操作####################################
    def get_depth_callback(self,data):
        """
        更新self.depth_image
        @param data:
        @return:
        """
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,'32FC1')#深度图获取到的单位是m位置度
        except CvBridgeError as e:
            print("[Error] Can't get depth_image",e)
            return
        # cv_image[np.isnan(cv_image)]=0
        self.depth_image=cv_image*10000
        self.depth_image[np.isnan(self.depth_image)]=0

    def read_image_depth(self,event,x,y,flags,param):
        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.depth_image[y,x]
                print("depth is :",hsv)
            except:
                print("maybe outof index")

    def get_color_callback(self,data):
        """
        更新self.color_image
        @param data:
        @return:
        """
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print("[Error] Can't get color_image",e)
            return
        self.color_image=cv_image

    def get_images(self):
        #这里面可以直接订阅两个函数
        depth_sub=rospy.Subscriber("/kinect/depth/image_raw",Image,self.get_depth_callback)
        image_sub=rospy.Subscriber("/kinect/color/image_raw",Image,self.get_color_callback)

    def show_image(self):
        if self.color_image is None or self.depth_image is None:
            print("[Warning] images is None")
        else:
            pass
            # cv.imshow("color_image",self.color_image)
            # cv.imshow("depth_image",self.depth_image)

    def process_image(self):
        """
        这里面主要针对color_image进行图片处理
        @return: 最终返回这个图片所对应的mask
        """
        #1:更新all_index内容
        self.get_all_index()
        mask=np.zeros(self.depth_image.shape,dtype=np.uint8)

        #2:解析对应的scene文件,获取对应的hsv,进行mask分割
        #2.1:寻找场景中所有的object_names
        world_object_list=[]
        worldfile_path=self.get_worldfile_path()
        xml_tree=ET.parse(worldfile_path).getroot()
        for child_1 in xml_tree.findall('world/model'):
            gazebo_name=child_1.attrib['name']
            if gazebo_name == "ground_plane" or gazebo_name == "table":
                continue
            for child_2 in child_1.findall('link/collision/geometry/mesh/uri'):
                uri=child_2.text
                uri=uri[8:]
                stop=uri.find('/')
                world_object_list.append(uri[:stop])
                break

        #2.2:根据所有的object_name,采用hsv分割得到目标的Mask,一张全0的Mask进行相加即可
        hsv_image=cv.cvtColor(self.color_image,cv.COLOR_BGR2HSV)
        self.hsv_image=hsv_image
        for object_in_world in world_object_list:
            hsv=self.all_index[object_in_world]['hsv']
            low=np.array([max(0,hsv[0]-5),max(0,hsv[1]-30),0])
            high=np.array([min(255,hsv[0]+5),min(255,hsv[1]+30),255])

            temp_mask=cv.inRange(hsv_image,low,high)
            mask_value=self.objects_names.index(object_in_world)+1
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
        print("already save image in ",self.dataset_pth+"/image_{}.png".format(sys.argv[0]))

        cv.imwrite(self.dataset_pth+"/image_{}.png".format(sys.argv[1]),color_map)
        return mask

    def make_data(self,hsv_mode=True,debug=False):
        """
        这里面生成所有需要的图片和矩阵信息
        @param hsv_mode:True:保存Mask,depth,color,False:保存color
        @param debug:
        @return:
        """
        #1:更新all_index内容
        self.get_all_index()
        mask=np.zeros(self.depth_image.shape,dtype=np.uint8)

        #2:解析对应的scene文件,获取对应的hsv,进行mask分割
        #2.1:寻找场景中所有的object_names
        world_object_list=[]
        worldfile_path=self.get_worldfile_sys_path()
        xml_tree=ET.parse(worldfile_path).getroot()
        for child_1 in xml_tree.findall('world/model'):
            gazebo_name=child_1.attrib['name']
            if gazebo_name == "ground_plane" or gazebo_name == "table":
                continue
            for child_2 in child_1.findall('link/collision/geometry/mesh/uri'):
                uri=child_2.text
                uri=uri[8:]
                stop=uri.find('/')
                world_object_list.append(uri[:stop])
                break

        #2.2:根据所有的object_name,采用hsv分割得到目标的Mask,一张全0的Mask进行相加即可
        hsv_image=cv.cvtColor(self.color_image,cv.COLOR_BGR2HSV)
        self.hsv_image=hsv_image
        for object_in_world in world_object_list:
            hsv=self.all_index[object_in_world]['hsv']
            low=np.array([max(0,hsv[0]-5),max(0,hsv[1]-30),0])
            high=np.array([min(255,hsv[0]+5),min(255,hsv[1]+30),255])

            temp_mask=cv.inRange(hsv_image,low,high)
            mask_value=self.objects_names.index(object_in_world)+1
            temp_mask=temp_mask/255*mask_value
            temp_mask=temp_mask.astype(np.uint8)
            mask=mask+temp_mask

        #2.3:进行Mask的展示
        mask=mask.astype(np.uint8)
        if debug:
            color_map=self.depth_image.copy()
            cv.normalize(color_map,color_map,255,0,cv.NORM_MINMAX)
            color_map=color_map.astype(np.uint8)
            color_map=cv.applyColorMap(color_map,cv.COLORMAP_JET)
            cv.namedWindow("color_map",cv.WINDOW_NORMAL)
            cv.imshow("color_map",color_map)
            cv.namedWindow("hsv_image",cv.WINDOW_NORMAL)
            cv.imshow("hsv_image",hsv_image)
            cv.imshow("depth_image",self.depth_image)
            cv.setMouseCallback("depth_image",self.read_image_depth)
            cv.imshow("mask",mask)
            print("world_object_list:",world_object_list)
            print("each hsv index:")
            for object_in_world in world_object_list:
                hsv=self.all_index[object_in_world]['hsv']
                low=np.array([max(0,hsv[0]-5),max(0,hsv[1]-30),0])
                high=np.array([min(255,hsv[0]+5),min(255,hsv[1]+30),255])
                print("object:{},low:{},high:{}".format(object_in_world,low,high))
            cv.waitKey(0)
        else:
            print("\n\nHSV Mode:{}\n already save scene_id:{}\n\n".format(hsv_mode,sys.argv[1]))
            if hsv_mode:
                #hSVMode中只产生depth.png,label.png
                depth_path=self.dataset_pth+"/{}-depth.png".format(sys.argv[1])
                with open(depth_path,'wb') as f:
                    writer=png.Writer(width=self.depth_image.shape[1],height=self.depth_image.shape[0],bitdepth=16)
                    zgray2list=self.depth_image.tolist()
                    writer.write(f,zgray2list)

                cv.imwrite(self.dataset_pth+"/{}-label.png".format(sys.argv[1]),mask)
            else:
                #非HSVMode中只产生color.png
                cv.imwrite(self.dataset_pth+"/{}-color.png".format(sys.argv[1]),self.color_image)
            sys.exit()
        return mask

def test_read_images():
    rospy.init_node("Make_Data")
    make_Data=Make_Data()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        make_Data.get_images()
        if make_Data.color_image is not None and make_Data.depth_image is not None:
            make_Data.make_data(hsv_mode=True,debug=False)
        else:
            print('[Warning] color_image not exist')
        rate.sleep()

def test_read_info():
    make_Data=Make_Data()
    make_Data.get_all_index()

def test_diff_world():
    make_Data=Make_Data()
    make_Data.change_world()

def makedata(HSV_MODE=True):
    """
    制作数据的核心函数,HSV模式下,生成图片和meta信息
    非HSV模式下,只生成RGB图片
    @param HSV_MODE:
    @return:
    """
    #1:初始化各种类
    rospy.init_node("Make_Data")
    read_Data=Read_Data.Read_Data()
    make_Data=Make_Data()
    rate = rospy.Rate(10)

    if HSV_MODE:
        #2:生成meta信息
        #2.1:获取各个物体的Pose和相机世界坐标系的变换
        world_info_list=read_Data.get_world_info(scene_id=sys.argv[1])
        # Trans_world2Camera=read_Data.get_T_Matrix()
        Trans_world2Camera=np.array([[-0.6184985,-0.14367474,-0.77253944,0.8138817 ],
                                     [ 0.12796457,-0.98843452,0.08137731,0.01304234],
                                    [-0.77529651,-0.04852593 ,0.62973054  ,0.07146605],
                                [ 0.   ,       0.  ,        0.   ,       1.        ]])
        cls_indexes=[]
        poses=[]
        factor_depth=[10000]
        intrinsic_matrix=make_Data.Camera_Info
        for each_object in world_info_list:
            true_name=each_object['true_name']
            class_id=make_Data.objects_names.index(true_name)+1
            model_pose=each_object['model_pose']

            #继续获取他的对应4D姿态
            Trans_raw2world=read_Data.get_matrix_from_modelpose(model_pose)
            Trans_all=Trans_world2Camera.dot(Trans_raw2world)

            cls_indexes.append(class_id)
            poses.append(Trans_all[0:3,:])

        poses=np.array(poses)
        poses=np.transpose(poses,(1,2,0))
        save_dict={}
        save_dict['cls_indexes']=cls_indexes
        save_dict['factor_depth']=factor_depth
        save_dict['intrinsic_matrix']=intrinsic_matrix
        save_dict['poses']=poses
        scio.savemat(make_Data.dataset_pth+"/{}-meta".format(sys.argv[1]),save_dict)

        #3:保存颜色图,深度图,label图
        while not rospy.is_shutdown():
            make_Data.get_images()
            if make_Data.color_image is not None and make_Data.depth_image is not None:
                make_Data.make_data(hsv_mode=HSV_MODE,debug=False)
            else:
                print('[Warning] color_image not exist')
            rate.sleep()
    else:
        #3:保存颜色图,深度图,label图
        while not rospy.is_shutdown():
            make_Data.get_images()
            if make_Data.color_image is not None and make_Data.depth_image is not None:
                make_Data.make_data(hsv_mode=HSV_MODE,debug=False)
            else:
                print('[Warning] color_image not exist')
            rate.sleep()

def check_makecorrect():
    read_data=Read_Data.Read_Data()
    make_data=Make_Data()

    index=sys.argv[1]
    meta = scio.loadmat(make_data.dataset_pth+'/{}-meta.mat'.format( index))
    poses=meta['poses']
    print(poses.shape)
    cls_indexes=meta['cls_indexes']

    show_points=[]
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    show_points.append(axis_point)
    for i,class_id in enumerate(cls_indexes[0]):
        true_name=make_data.objects_names[class_id-1]
        temp=np.zeros((4,4))
        pose=poses[:,:,i]
        temp[0:3,:]=pose
        temp[3,:]=[0,0,0,1]
        points=read_data.read_YCB.objects_points[true_name]
        target_o3d=o3d.geometry.PointCloud()
        target_o3d.points=o3d.utility.Vector3dVector(points)
        target_o3d.transform(temp)
        show_points.append(target_o3d)

    o3d.visualization.draw_geometries(show_points)


if __name__ == '__main__':
    # check_makecorrect()
    if sys.argv[2] is not None:
        if sys.argv[2]=='True':
            makedata(HSV_MODE=True)#一定要记得改models
        elif sys.argv[2]=='False':
            makedata(HSV_MODE=False)#一定要记得改models
        else:
            print("The second input is None,exit()")
            sys.exit()
    else:
        makedata(HSV_MODE=False)#一定要记得改models
    # test_diff_world()
    # test_read_images()
    # test_read_info()