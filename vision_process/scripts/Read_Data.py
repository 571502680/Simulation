#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
这个类中用于获取各种各样的数据信息,包含图片信息(包含HSV信息),深度图等等

Log:2020.9.10:
    完成点云处理部分代码,目前Pose已经能够正常解析,Scale等的问题也算是解决.注意这里面的obj文件是Collision中的obj,而非Meshes中的obj

Log:2020.9.24:
    加入基于Gazebo的物体读取和基于Sapien的物体读取,另外整合API接口,这里面主要就是负责读取各种信息的,MakeData中负责进行文件的生成
"""
import sys
import os
import rospy
import cv2 as cv
import numpy as np
import math
import time
import tf
import tf.transformations as trans_tools
import open3d as o3d
import xml.etree.ElementTree as ET
#ROS中的通信协议
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image,CameraInfo
from gazebo_msgs.srv import GetModelState,SetModelState,GetModelStateRequest,GetWorldProperties,GetWorldPropertiesRequest
import traceback
python_path=os.path.dirname(__file__)

class Read_YCB:
    """
    这里面用于读取YCB格式的数据集,然后生成对应的点
    """
    def __init__(self,get_object_points=False):
        self.classes_file_path=python_path+"/classes.txt"
        self.objects_points={}
        self.objects_names=self.get_objects_names()
        if get_object_points:
            self.get_objects_points()#这里进行self.objects_points的填充

    def get_objects_names(self):
        """
        这里面进行此目录中classes.txt的文件更新
        :return:
        """
        file=open(python_path+"/classes.txt",'r')
        class_names=file.readlines()
        objects_name=[]
        for class_name in class_names:
            class_name=class_name[:-1]#去掉\n
            objects_name.append(class_name)
        return objects_name

    def get_objects_points(self):
        """
        基于class_file中的所有类别,读取对应类别的points.xyz文件
        @return:
        """
        for object_name in self.objects_names:
            points_file=open(python_path+'/object_models/{}/points.xyz'.format(object_name))
            self.objects_points[object_name] = []
            while 1:
                input_line = points_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.objects_points[object_name].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.objects_points[object_name] = np.array(self.objects_points[object_name])
            points_file.close()

    def read_obj_from_file(self,obj_path,see_file=False):
        """
        (弃用)
        通过解析obj源文件,对里面的v开头的行进行解析,读取一个obj文件,返回N*3个点的np矩阵
        @param obj_path:obj文件路径
        @param see_file:是否进行点云可视化
        @return:返回这个文件对应的点云
        """
        data=open(obj_path)
        points=[]

        for line in data.readlines():
            if line.startswith('#'):
                continue
            values=line.split()
            if not values:#如果没有值(空行),也跳过
                continue

            if values[0]=='v':
                v=[ float(x) for x in values[1:4]]
                points.append(v)

        points=np.array(points)
        if see_file:
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])#可视化

        return points

class Read_Data:
    def __init__(self,init_node=False,read_images=True,simulator="sapien"):
        if init_node:
            rospy.init_node("Read_Data")

        #读取图片信息
        self.bgr_image=None
        self.depth_image=None
        self.camera_info=None
        #默认好的摄像头信息
        self.Camera_Info=np.array([[1120.1199067175087, 0.0, 640.5], [0.0, 1120.1199067175087], [360.5, 0.0, 0.0, 1.0]])
        self.bridge=CvBridge()
        if read_images:
            self.begin_get_depth()
            self.begin_get_image()

        #仿真器
        self.simulator=simulator

        #指定读取路径
        ##导入python路径
        self.python_path=os.path.dirname(__file__)
        self.materials_path=rospy.get_param('~materials_dir','/root/ocrtoc_materials')#默认是root路径下

        #导入点云文件(暂时先保存一个)
        self.save_image_flag=False

        #read_image_bgr
        self.read_image_bgr=None
        self.read_image_hsv=None
        self.last_hsv=None

        if simulator=="sapien":
            self.Trans_camera2world=np.array( [[-0.6184985 , 0.12796457,-0.77529651,0.55712303],
                                               [-0.14367474,-0.98843452,-0.04852593,0.13329369],
                                               [-0.77253944, 0.08137731, 0.62973054,0.58269   ],
                                               [ 0.        , 0.        , 0.        ,1.        ]])

            self.Trans_world2camera=np.array(  [[-0.6184985 ,-0.14367474,-0.77253944,0.8138817 ],
                                                [ 0.12796457,-0.98843452, 0.08137731,0.01304234],
                                                [-0.77529651,-0.04852593, 0.62973054,0.07146605],
                                                [ 0.        , 0.        , 0.        ,1.        ]])

        elif simulator=="gazebo":
            self.Trans_camera2world=np.array([[-0.6184985 ,  0.12796457, -0.77529651,  0.55712303],
                                              [-0.14367474, -0.98843452, -0.04852593,  0.13329369],
                                              [-0.77253944,  0.08137731,  0.62973054,  0.58269   ],
                                              [ 0.        ,  0.        ,  0.        ,  1.        ]])

            self.Trans_world2camera=np.array(  [[-0.6184985 ,-0.14367474,-0.77253944,0.8138817 ],
                                                [ 0.12796457,-0.98843452, 0.08137731,0.01304234],
                                                [-0.77529651,-0.04852593, 0.62973054,0.07146605],
                                                [ 0.        , 0.        , 0.        ,1.        ]])


    ####################################读取图片的函数##################################
    def depth_process_callback(self,data,see_image=False):
        """
        深度图像处理
        :param data:深度图像
        :param see_image: 是否可视化图片
        :return:
        """
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,'32FC1')
            self.depth_image=cv_image*10000
            self.depth_image[np.isnan(self.depth_image)]=0
        except CvBridgeError as e:
            print("[Error] depth_process_callback occur error {}".format(e))
            return

        if see_image:
            if self.depth_image is  None:
                print("[Warning] Can not get the depth_image")
                return
            cv_image=self.depth_image.copy()
            cv_image=cv_image*1000
            ROI=cv.inRange(cv_image,0,1500)#对于太远的去掉,省得看的并不明显
            cv_image=cv_image*ROI/255
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
            cv.namedWindow("color_map",cv.WINDOW_NORMAL)
            cv.imshow("color_map",color_map)
            cv.waitKey(3)

    def read_image_hsv_callback(self,event,x,y,flags,param):
        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.read_image_hsv[y,x]
                print("HSV is :",hsv)
            except:
                print("[Warning] the target is output range")

    def read_image_bgr_callback(self,event,x,y,flags,param):
        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.read_image_bgr[y,x]
                print("BGR is :",hsv)
            except:
                print("[Warning] the target is output range")

    def image_process_callback(self,data,see_image=False,read_HSV=False):
        """
        RGB图可视化
        :param data:rgb图像
        :param see_image: 是否查看图片
        :param read_HSV: 是否读取HSV点
        :return:
        """
        try:
            self.bgr_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print("[Error] image_process_callback occur error {}".format(e))
            return

        if read_HSV and not see_image:
            print("[Warning] if you want to read HSV data,please set see_image True")

        if see_image:
            if self.bgr_image is  None:
                print("[Warning] Can not get the bgr_image")
                return
            cv_image=self.bgr_image.copy()
            self.read_image_bgr=cv_image
            cv.namedWindow("image_get",cv.WINDOW_NORMAL)
            cv.imshow("image_get",cv_image)
            cv.setMouseCallback("image_get",self.read_image_bgr_callback)
            if read_HSV:
                hsv_image=cv.cvtColor(cv_image,cv.COLOR_BGR2HSV)
                self.read_image_hsv=hsv_image
                cv.namedWindow("hsv_get",cv.WINDOW_NORMAL)
                cv.imshow("hsv_get",hsv_image)
                cv.setMouseCallback("hsv_get",self.read_image_hsv_callback)
            cv.waitKey(3)

    def camera_info_callback(self,data):
        print("Camera Info is :\n",data)
        self.camera_info=data

    def begin_get_depth(self):
        callback_lambda=lambda x:self.depth_process_callback(x,see_image=False)
        depth_sub=rospy.Subscriber("/kinect/depth/image_raw",Image,callback_lambda)

    def begin_get_image(self):
        callback_lambda=lambda x:self.image_process_callback(x,see_image=False,read_HSV=False)
        image_sub=rospy.Subscriber("/kinect/color/image_raw",Image,callback_lambda)

    def begin_get_camera_info(self):
        kinect_info=rospy.Subscriber("/kinect/color/camera_info",CameraInfo,self.camera_info_callback)

    def get_images(self):
        if self.bgr_image is not None and self.depth_image is not None:
            return self.bgr_image,self.depth_image
        else:
            if self.bgr_image is None:
                print("[Warning] bgr_image is None")
            if self.depth_image is None:
                print("[Warning] depth_image is None")
            time.sleep(0.5)#避免输出太快
            return None,None

    ####################################读取Gazebo中物体Pose的函数##################################
    def get_matrix_from_modelpose(self,model_pose):
        """
        这里面从model_pose变换到矩阵
        是一个4*4矩阵
        @return:
        """
        t=np.array([model_pose.position.x,model_pose.position.y,model_pose.position.z])
        rot=[model_pose.orientation.x,model_pose.orientation.y,model_pose.orientation.z,model_pose.orientation.w]
        rotation_matrix=np.array(trans_tools.quaternion_matrix(rot))
        rotation_matrix[0:3,3]=t.T
        return rotation_matrix

    def get_object_pose_from_simulator(self,model_name):
        """
        这里面从self.simulator中获取物体的Pose
        :param model_name: sapien中物体的名称
        :return: 物体对应的xyz和四元素
        """
        #1:生成读取节点
        if self.simulator=='gazebo':
            get_object_service=rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        elif self.simulator=="sapien":
            get_object_service=rospy.ServiceProxy('/sapien/get_model_state',GetModelState)
        else:
            print("[Error] Please input the correct simulator")
            sys.exit()

        #2:获取物体的pose
        model=GetModelStateRequest()#生成一个request,srv说明中,model包含model_name
        model.model_name=model_name
        object_state=get_object_service(model)
        return object_state.pose

    def get_object_pose_from_worldfile(self,model_name,scene_id="1-1"):
        """
        获取所有物体的Pose(这个用处不大)
        :param model_name: 物体名称
        :param scene_id: 获取的对应场景
        :return:
        """
        worldfile_path=self.materials_path+"/scenes/"+scene_id+"/input.world"
        root=ET.parse(worldfile_path).getroot()
        var_list=None
        for child in root.findall('world/model'):
            object_name = child.attrib['name']
            if object_name!=model_name:
                continue
            for pose in child.findall('pose'):
                var_list = list(map(float, pose.text.split(' ')))
        if var_list is None:
            print("[Error] the function don't get a var_list")
        return var_list

    def get_world_info(self,scene_id="1-1",debug=False):
        """
        对模拟器中获取世界信息,无论Gazebo还是Sapien都可以使用这个进行获取
        进行.world文件的解析,获取其中的各种有用信息并进行保存
        (与Gazebo中不同的就是Sapien不可以直接获取所有物体的名称)
        :param debug:是否进行结果输出
        :param scene_id: 对应场景id
        :return: world_info_list:一个list,每个是一个dict,dict中包含了一个物体的gazebo_name,true_name,model_pose(model_pose是Pose的消息类型)
        :return: gazebo_name2true_name:从一个gazebo_name中索引到对应的真实物体名称
        """
        world_info_list=[]

        #1:获取所有物体的名称
        gazebo_name2true_name={}#保存成一个dict,由gazebo名称直接变换到真实名称
        gazebo_name_list=[]
        worldfile_path=self.materials_path+"/scenes/"+scene_id+"/input.world"
        xml_tree=ET.parse(worldfile_path).getroot()
        for child_1 in xml_tree.findall('world/model'):
            gazebo_name=child_1.attrib['name']
            if gazebo_name == "ground_plane" or gazebo_name == "table":
                continue

            gazebo_name_list.append(gazebo_name)
            for child_2 in child_1.findall('link/collision/geometry/mesh/uri'):
                uri=child_2.text
                uri=uri[8:]
                stop=uri.find('/')
                gazebo_name2true_name[gazebo_name]=uri[:stop]
                break

        #2:从仿真器中获取物体的pose
        for gazebo_name in gazebo_name_list:
            try:
                if gazebo_name=="ground" or gazebo_name=="robot":
                    continue
                model_pose=self.get_object_pose_from_simulator(gazebo_name)
                true_name=gazebo_name2true_name[gazebo_name]
                world_info_list.append({'gazebo_name':gazebo_name,'true_name':true_name,'model_pose':model_pose})
                if debug:
                    print("*"*50)
                    print("Model is :{}".format(gazebo_name))
                    print("True name is :{}".format(true_name))
                    print("Model Pose is : {}".format(model_pose))
            except:
                print('[Error] Can not find model name:{} in name_index'.format(gazebo_name))
                continue

        return world_info_list,gazebo_name2true_name,gazebo_name_list

    def get_T_Matrix(self,target_frame="base_link",source_frame="world"):
        """
        获取T_worldtoCamera
        @return:在world坐标系中,base_link的4*4齐次矩阵
        """
        camera_listener=tf.TransformListener()
        trans=None
        rot=None
        while not rospy.is_shutdown():
            try:
                trans,rot=camera_listener.lookupTransform(target_frame=target_frame,source_frame=source_frame,time=rospy.Time(0))
                break
            except :
                print("[Warning] get trans and rot failed!")
                time.sleep(0.5)
                continue

        # print("Get the Trans:{},rot{}".format(trans,rot))
        Trans_world2Camera=trans_tools.quaternion_matrix(rot)
        Trans_world2Camera[0:3,3]=np.array(trans).T#融合上T
        #获取opencv中的rect和trans的操作
        # rvec=cv.Rodrigues(Trans_world2Camera[0:3,0:3])[0]
        # print("rvec:",rvec)
        # print("tvec:",trans)

        return Trans_world2Camera

#################################测试函数##################################
def test_read_image():
    """
    用于测试读取图片函数
    @return:
    """
    read_Data=Read_Data(init_node=True)
    while not rospy.is_shutdown():
        bgr_image,depth_image=read_Data.get_images()
        if bgr_image is not None:
            cv.imshow("bgr_iamge",bgr_image)
            cv.imwrite("bgr_image.png",bgr_image)
            break

def test_get_camera_info():
    read_Data=Read_Data(init_node=True)
    read_Data.begin_get_camera_info()
    while not rospy.is_shutdown():
        if read_Data.camera_info is not None:
            break
        else:
            print("[Warning] Waiting for the Camera Info")
            time.sleep(0.5)

def test_get_object_pose():
    """
    用于测试读取model姿态代码
    @return:
    """
    read_Data=Read_Data(init_node=True)
    read_Data.get_world_info(debug=True)

def test_get_Trans_Matrix():
    """
    这里面获取整个物体的mask(相对于Kinect的)
    @return:
    """
    read_Data=Read_Data(init_node=True)
    Trans_Matrix=read_Data.get_T_Matrix(target_frame='kinect_camera_visor',source_frame='world')
    print("Tran_Matrix is:\n {}".format(Trans_Matrix))

def check_object_pose(debug=False):
    """
    这里面获取物体的Pose以及对应物体的xyz点,然后采用open3d查看物体是否是对应位置的
    @return:
    """
    read_Data=Read_Data(init_node=True)
    read_YCB=Read_YCB(get_object_points=True)
    #1:获取Gazebo中物体的Pose
    #获取world_info的dict,这里面有每一个物体的Pose,XML中读取的Pose,GazeboName和真实名称
    world_info_list,gazebo_name2true_name,gazebo_name_list=read_Data.get_world_info(debug=False)

    #生成需要产生的点云
    show_points=[]#用于保存所有需要展示的points
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    show_points.append(axis_point)

    #2:对每个物体变换到目标位置
    for i,object in enumerate(world_info_list):
        gazebo_name=object['gazebo_name']
        true_name=object['true_name']
        model_pose=object['model_pose']

        #获取目标点云
        points=read_YCB.objects_points[true_name]

        #xyz+rpy方式的Pose
        # t=np.array([model_pose[0],model_pose[1],model_pose[2]])
        # r=np.array([model_pose[3],model_pose[4],model_pose[5]])
        # rotation_matrix=trans_tools.euler_matrix(model_pose[3],model_pose[4],model_pose[5])
        # rotation_matrix[0:3,3]=t.T
        # print(rotation_matrix)

        #xyz+四元数方式的Pose
        t=np.array([model_pose.position.x,model_pose.position.y,model_pose.position.z])
        rot=[model_pose.orientation.x,model_pose.orientation.y,model_pose.orientation.z,model_pose.orientation.w]
        rotation_matrix=np.array(trans_tools.quaternion_matrix(rot))
        rotation_matrix[0:3,3]=t.T


        #直接读取points
        target_o3d=o3d.geometry.PointCloud()
        target_o3d.points=o3d.utility.Vector3dVector(points)
        target_o3d.transform(rotation_matrix)
        target_axis=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        target_axis.transform(rotation_matrix)
        show_points.append(target_o3d)
        show_points.append(target_axis)
        if debug:
            print("#"*50)
            print("add gazebo name:",gazebo_name)
            print("true name:",true_name)

    #2:将Gazebo中的Pose进行展示
    o3d.visualization.draw_geometries(show_points)

def get_pose_in_camera():
    """
    测试相机下Pose是正确的
    :return:
    """
    read_Data=Read_Data(init_node=True)
    read_YCB=Read_YCB(get_object_points=True)
    #获取物体在Base坐标系下的Pose
    world_info_list=read_Data.get_world_info()

    #变换到Camera坐标系下的Pose
    Trans_world2Camera=read_Data.get_T_Matrix(target_frame="kinect_camera_visor",source_frame="world")

    #直接所有物体变换到Camera坐标系下
    #1:先获取所有物体的点云
    show_pointscloud=[]
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    show_pointscloud.append(axis_point)
    for each_object in world_info_list:
        true_name=each_object['true_name']
        model_pose=each_object['model_pose']
        points=read_YCB.objects_points[true_name]#获取点云

        #继续获取他的对应4D姿态
        Trans_raw2world=read_Data.get_matrix_from_modelpose(model_pose)
        Trans_all=Trans_world2Camera.dot(Trans_raw2world)
        target_o3d=o3d.geometry.PointCloud()
        target_o3d.points=o3d.utility.Vector3dVector(points)
        target_o3d.transform(Trans_all)
        target_axis=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        target_axis.transform(Trans_all)

        #保存对应点云
        show_pointscloud.append(target_o3d)
        show_pointscloud.append(target_axis)

    #2:所有点云变换到Camera坐标系下
    o3d.visualization.draw_geometries(show_pointscloud)

#################################功能函数#############################


if __name__ == '__main__':
    test_get_Trans_Matrix()
    # test_read_image()
    # test_get_camera_info()
    # test_get_object_pose()
    # check_object_pose()
    # get_pose_in_camera()






