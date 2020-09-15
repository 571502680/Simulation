#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
这个类中用于获取各种各样的数据信息,包含图片信息(包含HSV信息),深度图等等

Log:2020.9.10:
    完成点云处理部分代码,目前Pose已经能够正常解析,Scale等的问题也算是解决.注意这里面的obj文件是Collision中的obj,而非Meshes中的obj


"""
import sys
import os
import rospy
import cv2 as cv
import numpy as np
import math
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
    def __init__(self):
        self.classes_file_path=python_path+"/classes.txt"
        self.root=python_path+"/../ALi_dataset"
        self.objects_points={}
        self.objects_names=['a_cups', 'a_lego_duplo', 'a_toy_airplane', 'adjustable_wrench', 'b_cups', 'b_lego_duplo', 'b_toy_airplane', 'banana', 'bleach_cleanser', 'bowl', 'bowl_a', 'c_cups', 'c_lego_duplo', 'c_toy_airplane', 'cracker_box', 'cup_small', 'd_cups', 'd_lego_duplo', 'd_toy_airplane', 'e_cups', 'e_lego_duplo', 'e_toy_airplane', 'extra_large_clamp', 'f_cups', 'f_lego_duplo', 'flat_screwdriver', 'foam_brick', 'fork', 'g_cups', 'g_lego_duplo', 'gelatin_box', 'h_cups', 'hammer', 'i_cups', 'j_cups', 'jenga', 'knife', 'large_clamp', 'large_marker', 'master_chef_can', 'medium_clamp', 'mug', 'mustard_bottle', 'nine_hole_peg_test', 'pan_tefal', 'phillips_screwdriver', 'pitcher_base', 'plate', 'potted_meat_can', 'power_drill', 'prism', 'pudding_box', 'rubiks_cube', 'scissors', 'spoon', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can', 'wood_block']
        self.get_objects_points(self.classes_file_path)#这里进行self.objects_points的填充

    def read_obj_from_file(self,obj_path,see_file=False):
        """
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

    def get_objects_points(self,classes_file_path):
        """
        基于class_file中的所有类别,读取对应类别的points.xyz文件
        @param classes_file_path: classes.txt文件路径
        @return:
        """
        class_file = open(classes_file_path)
        while 1:
            class_input = class_file.readline()
            object_name=class_input[:-1]
            if not class_input:
                break
            points_file = open('{0}/models/{1}/collision_meshes/points.xyz'.format(self.root,object_name))
            self.objects_points[object_name] = []
            while 1:
                input_line = points_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.objects_points[object_name].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.objects_points[object_name] = np.array(self.objects_points[object_name])
            points_file.close()

    def get_objects_points_from_obj(self,classes_file_path):
        """
        基于classes_file更新self.ojbects_points,但是这里面保存的是mesh文件,不是np.array文件
        @param classes_file_path:
        @return:
        """
        class_file = open(classes_file_path)
        while 1:
            class_input = class_file.readline()

            if not class_input:
                break
            points=o3d.io.read_triangle_mesh('{0}/models/{1}/collision_meshes/collision.obj'.format(self.root, class_input[:-1]))

            self.objects_points[class_input[:-1]]=points

class Read_Data:
    def __init__(self):
        self.bridge=CvBridge()
        self.Get_T_Matrix_Flag=False
        self.Camera_Info=np.array([[1120.1199067175087, 0.0, 640.5], [0.0, 1120.1199067175087], [360.5, 0.0, 0.0, 1.0]])

        #导入点云文件(暂时先保存一个)
        self.save_image_flag=False

        #read_image_rgb
        self.read_image_rgb=None
        self.last_hsv=None

        #导入自己数据集的点
        self.read_YCB=Read_YCB()

        #导入python路径
        self.python_path=os.path.dirname(__file__)

    ####################################读取图片的函数##################################
    def depth_process_callback(self,data):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,'32FC1')#深度图获取到的单位是m位置度
        except CvBridgeError as e:
            print("error:",e)
            return

        cv_image=cv_image*1000
        ROI=cv.inRange(cv_image,0,1500)#对于太远的去掉,省得看的并不明显
        cv_image=cv_image*ROI/255
        cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
        cv_image=cv_image.astype(np.uint8)
        color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
        cv.namedWindow("color_map",cv.WINDOW_NORMAL)
        cv.imshow("color_map",color_map)

        cv.waitKey(3)

    def read_image_hsv(self,event,x,y,flags,param):
        if event==cv.EVENT_MOUSEMOVE:
            hsv=self.read_image_rgb[y,x]
            print("HSV is :",hsv)

    def image_process_callback(self,data):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")

        except CvBridgeError as e:
            print(e)
            return

        cv.namedWindow("image_get",cv.WINDOW_NORMAL)
        cv.imshow("image_get",cv_image)
        hsv_image=cv.cvtColor(cv_image,cv.COLOR_BGR2HSV)
        self.read_image_rgb=hsv_image
        cv.namedWindow("hsv_get",cv.WINDOW_NORMAL)
        cv.imshow("hsv_get",hsv_image)
        cv.setMouseCallback("hsv_get",self.read_image_hsv)
        if self.save_image_flag:
            self.save_image_flag=False
            cv.imwrite("/root/ocrtoc_ws/src/vision_process/scripts/temp.png",cv_image)

        temp=hsv_image[:,:,0]
        cv.imshow("temp",temp)
        cv.waitKey(3)

    def get_depth(self):
        depth_sub=rospy.Subscriber("/kinect/depth/image_raw",Image,self.depth_process_callback)

    def get_image(self):
        image_sub=rospy.Subscriber("/kinect/color/image_raw",Image,self.image_process_callback)

    def camera_info_callback(self,data):
        print("Camera Info is :\n",data)

    def get_camera_info(self):
        kinect_info=rospy.Subscriber("/kinect/color/camera_info",CameraInfo,self.camera_info_callback)

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

    def get_object_pose(self,model_name):
        """
        这里面从Gazebo中获取物体的Pose
        @return:物体对应的xyz和四元素
        """
        get_object_service=rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model=GetModelStateRequest()#生成一个request,srv说明中,model包含model_name
        model.model_name=model_name
        object_state=get_object_service(model)
        return object_state.pose

    def get_object_pose_from_xml(self,model_name):
        task_path=self.get_worldfile_path()
        root=ET.parse(task_path).getroot()
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

    def get_worldfile_path(self,scene_id=None):
        """
        这里面用于读取world的tree
        @return:
        """
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

    def get_world_info(self,debug=False,scene_id=None):
        """
        获取世界中的物体信息
        这里面还需要加上一个名称和实际物体的解析代码
        @return:
        """
        world_info_list=[]
        #1: 获取所有model的名称,另外获取对应world路径
        get_object_service=rospy.ServiceProxy('/gazebo/get_world_properties',GetWorldProperties)
        request=GetWorldPropertiesRequest()#此request是空的,srv中的说明里面这里也是空的
        world_info=get_object_service(request)
        model_names=world_info.model_names

        #2:解析出每个gazebo_name对应的true_name
        #读取xml文件
        if scene_id is not None:
            worldfile_path=self.get_worldfile_path(scene_id)
        else:
            worldfile_path=self.get_worldfile_path()
        xml_tree=ET.parse(worldfile_path).getroot()
        name_index={}#保存成一个dict
        for child_1 in xml_tree.findall('world/model'):
            gazebo_name=child_1.attrib['name']
            if gazebo_name == "ground_plane" or gazebo_name == "table":
                continue
            for child_2 in child_1.findall('link/collision/geometry/mesh/uri'):
                uri=child_2.text
                uri=uri[8:]
                stop=uri.find('/')
                name_index[gazebo_name]=uri[:stop]
                break

        #3:获取除了robot和ground以外的物体的Pose(相对于world的)
        for model_name in model_names:
            try:
                if model_name=="ground" or model_name=="robot":
                    continue
                model_pose=self.get_object_pose(model_name)
                # model_pose=self.get_object_pose_from_xml(model_name)
                true_name=name_index[model_name]
                world_info_list.append({'gazebo_name':model_name,'true_name':true_name,'model_pose':model_pose})
                if debug:
                    print("*"*50)
                    print("Model is :\n",model_name)
                    print("True name is :\n",true_name)
                    print("Model Pose is :\n",model_pose)

            except:
                print('[Error] Can not find model name:{} in name_index'.format(model_name))
                continue

        return world_info_list

    def get_T_Matrix(self):
        """
        获取T_worldtoCamera
        @return:
        """
        camera_listener=tf.TransformListener()
        trans=None
        rot=None
        while not rospy.is_shutdown():
            try:
                trans,rot=camera_listener.lookupTransform('kinect_camera_visor','world',rospy.Time(0))#获取到两个坐标系之间的关系
                self.Get_T_Matrix_Flag=True
            except :
                print("[Warning] get trans and rot failed!")
                continue
            print("Get the Trans:{},rot{}".format(trans,rot))

            if self.Get_T_Matrix_Flag:
                break

        Trans_world2Camera=trans_tools.quaternion_matrix(rot)
        Trans_world2Camera[0:3,3]=np.array(trans).T#融合上T

        rvec=cv.Rodrigues(Trans_world2Camera[0:3,0:3])[0]
        print("rvec:",rvec)
        print("tvec:",trans)

        return Trans_world2Camera

#################################测试函数##################################
def test_read_image():
    """
    用于测试读取图片函数
    @return:
    """
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    try:
        read_Data.get_depth()
        read_Data.get_image()
        rospy.spin()
    except KeyboardInterrupt:
        print("Stop to Get Data")
        cv.destroyAllWindows()

def test_get_object_pose():
    """
    用于测试读取model姿态代码
    @return:
    """
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    read_Data.get_world_info()

def test_get_Trans_Matrix():
    """
    这里面获取整个物体的mask(相对于Kinect的)
    @return:
    """
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    Trans_Matrix=read_Data.get_T_Matrix()
    print("Tran_Matrix is:\n",Trans_Matrix)

def test_get_camera_info():
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    read_Data.get_camera_info()
    rospy.spin()

def check_object_pose():
    """
    这里面获取物体的Pose以及对应物体的xyz点,然后采用open3d查看物体是否是对应位置的
    @return:
    """
    print("~~~"*60)
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    #1:获取Gazebo中物体的Pose
    #获取world_info的dict,这里面有每一个物体的Pose,XML中读取的Pose,GazeboName和真实名称
    world_info_list=read_Data.get_world_info(debug=False)

    #生成需要产生的点云
    show_points=[]#用于保存所有需要展示的points
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    show_points.append(axis_point)

    #2:对每个物体变换到目标位置
    for i,object in enumerate(world_info_list):
        print("#"*50)
        gazebo_name=object['gazebo_name']
        true_name=object['true_name']
        model_pose=object['model_pose']

        #获取目标点云
        points=read_Data.read_YCB.objects_points[true_name]


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

        show_points.append(target_o3d)
        print("add gazebo name:",gazebo_name)
        print("true name:",true_name)

    #2:将Gazebo中的Pose进行展示
    o3d.visualization.draw_geometries(show_points)

def check_pose_mesh():
    """
    这里面获取物体的Pose以及对应物体的xyz点,然后采用open3d查看物体是否是对应位置的
    @return:
    """
    print("~~~"*60)
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    #1:获取Gazebo中物体的Pose
    #获取world_info的dict,这里面有每一个物体的Pose,XML中读取的Pose,GazeboName和真实名称
    world_info_list=read_Data.get_world_info(debug=False)

    #生成需要产生的点云
    show_points=[]#用于保存所有需要展示的points
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)#添加中点坐标系
    show_points.append(axis_point)

    #2:对每个物体变换到目标位置
    for i,object in enumerate(world_info_list):
        print("#"*50)
        gazebo_name=object['gazebo_name']
        true_name=object['true_name']
        model_pose=object['model_pose']

        try:
            points=read_Data.read_YCB.objects_points[true_name]

            #2.2:xyz+四元数方式的Pose
            print(":!!!!!!!!!!!!!!!!!!!!!!!!Model_Pose:!!!!!!!!!!!!!!!!!!!!!!!!")
            print(model_pose)
            t=np.array([model_pose.position.x,model_pose.position.y,model_pose.position.z])
            rot=[model_pose.orientation.x,model_pose.orientation.y,model_pose.orientation.z,model_pose.orientation.w]
            rotation_matrix=np.array(trans_tools.quaternion_matrix(rot))
            rotation_matrix[0:3,3]=t.T
            print("Pose_Matrix:",rotation_matrix)


            #读取mesh
            points.transform(rotation_matrix)

            show_points.append(points)
            print("add gazebo name:",gazebo_name)
            print("true name:",true_name)

        except:
            print("[Warning] {} mesh not exist".format(true_name))
            print(traceback.format_exc())
            continue

    #2:将Gazebo中的Pose进行展示
    o3d.visualization.draw_geometries(show_points)

def get_pose_in_camera():
    rospy.init_node("Read_Data")
    read_Data=Read_Data()
    #获取物体在Base坐标系下的Pose
    world_info_list=read_Data.get_world_info()

    #变换到Camera坐标系下的Pose
    Trans_world2Camera=read_Data.get_T_Matrix()

    #直接所有物体变换到Camera坐标系下
    #1:先获取所有物体的点云
    show_pointscloud=[]
    axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    show_pointscloud.append(axis_point)
    for each_object in world_info_list:
        true_name=each_object['true_name']
        gazebo_name=each_object['gazebo_name']
        model_pose=each_object['model_pose']
        points=read_Data.read_YCB.objects_points[true_name]#获取点云

        #继续获取他的对应4D姿态
        Trans_raw2world=read_Data.get_matrix_from_modelpose(model_pose)
        Trans_all=Trans_world2Camera.dot(Trans_raw2world)
        target_o3d=o3d.geometry.PointCloud()
        target_o3d.points=o3d.utility.Vector3dVector(points)
        target_o3d.transform(Trans_all)

        #保存对应点云
        show_pointscloud.append(target_o3d)

    #2:所有点云变换到Camera坐标系下
    o3d.visualization.draw_geometries(show_pointscloud)

#################################功能函数#############################


if __name__ == '__main__':
    # get_pose_in_camera()
    # check_object_pose()
    # test_read_image()
    test_get_Trans_Matrix()
    # read_obj=Read_Obj()
    # read_obj.read_file("/home/elevenjiang/Desktop/ocrtoc_materials/models/a_cups/meshes/textured.obj")





