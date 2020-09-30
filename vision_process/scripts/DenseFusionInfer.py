#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
这里封装DenseFusion的使用函数,从而让整体可以前向推理
Log:2020.9.14:
    开始进行DenseFusion封装

Log:2020.9.15:
    整合了代码,避免代码太过复杂.
    SegNet采用np.argmax进行统计,与此同时,设置了一定的区域分割,从而确保没有太多其他参数

Log:2020.2.27:
    基于更改过的Read_Data和Make_Data,进行DenseFusion的执行

"""

import argparse
import copy
import sys
import os
import cv2 as cv
import numpy as np
import open3d as o3d
import numpy.ma as ma
import scipy.io as scio
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Int8
from gazebo_msgs.msg import ModelStates
import tf.transformations as trans_tools
import math

import Make_Data
import Read_Data
#DenseFusion库
from DenseFusion_Lib.network import PoseNet,PoseRefineNet
from DenseFusion_Lib.transformations import quaternion_matrix,quaternion_from_matrix,euler_from_quaternion
#SegNet库
from SegNet_Lib.segnet import SegNet

class DenseFusion_Detector:
    def __init__(self,model_path=None,refine_model_path=None,segnet_path=None,init_node=False):
        if init_node:
            rospy.init_node("DenseFusionInfer")
        #1:导入网络
        if model_path is None:
            self.model_path="DenseFusion_Lib/models/sapien_posemodel_0.030.pth"
        else:
            self.model_path=model_path

        if refine_model_path is None:
            self.refine_model_path="models/temp.pth"
        else:
            self.refine_model_path=refine_model_path
        self.python_path=os.path.dirname(__file__)+"/"
        self.num_points=1000
        self.num_obj=79
        self.estimator = PoseNet(num_points=self.num_points, num_obj=self.num_obj)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(self.python_path+self.model_path))
        self.estimator.eval()

        # refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
        # refiner.cuda()
        # refiner.load_state_dict(torch.load(opt.refine_model))
        # refiner.eval()

        #2:一系列需要的参数
        #图片变换参数
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.iteration=0#不进行迭代
        self.bs=1

        #图像参数
        self.img_width=720
        self.img_length=1280
        self.xmap = np.array([[j for i in range(self.img_length)] for j in range(self.img_width)])
        self.ymap = np.array([[i for i in range(self.img_length)] for j in range(self.img_width)])
        self.cam_cx =  640.5
        self.cam_cy =  360.5
        self.cam_fx =  1120.1199067175087
        self.cam_fy =  1120.1199067175087
        self.camera_matrix=np.array([[self.cam_cx,0,self.cam_fx],[0,self.cam_cy,self.cam_fy],[0,0,1]])
        self.dist=np.array([0,0,0,0,0],dtype=np.float)
        self.cam_scale=10000

        #3:导入所需点云模型
        class_file = open(self.python_path+'classes.txt')  # 读取一共包含的类别信息
        class_id = 1
        self.object_points = {}#保存物体的点云
        self.object_names=[]#保存问题的名称
        self.object_names.append('noname')#插入第一个,让class_id从1开始
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_input = class_input[:-1]  # 应该用于去掉换行符号
            self.object_names.append(class_input)#插入类别名称
            # 得到对应的model
            input_file = open(self.python_path+'object_models/{0}/points.xyz'.format(class_input))  # 得到他们对应的xyz文件
            self.object_points[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1]
                input_line = input_line.split(' ')
                self.object_points[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])  #获取他们的xyz
            input_file.close()
            self.object_points[class_id] = np.array(self.object_points[class_id])  # 将一个list变换为一个np.array
            class_id += 1

        #4:导入SegNet
        self.segnet=SegNet(input_nbr=3,label_nbr=80)
        if segnet_path is None:
            self.segnet_path="SegNet_Lib/models/segnet_sapien_0.021.pth"
        else:
            self.segnet_path=segnet_path
        # self.segnet.load_state_dict(torch.load(self.python_path+self.segnet_path))
        self.segnet.load_state_dict(torch.load(self.python_path+self.segnet_path))
        self.segnet=self.segnet.cuda()
        self.norm_seg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #5:坐标系变换
        self.Trans_camera2world=np.array([[-0.6184985 ,  0.12796457, -0.77529651,  0.55712303],
                                          [-0.14367474, -0.98843452, -0.04852593,  0.13329369],
                                          [-0.77253944,  0.08137731,  0.62973054,  0.58269   ],
                                          [ 0.        ,  0.        ,  0.        ,  1.        ]])



        #6:信息发布
        self.poses_pub=rospy.Publisher("/DenseFusionInfer/PoseinWorld",ModelStates,queue_size=10)
        self.detect_state_sub=rospy.Subscriber("/DenseFusionInfer/DetectState",Int8,self.update_detect_state)
        self.DETECT_FLAG=True#用于重新检测物体姿态
        self.STOP_FLAG=False#用于停止程序

    ###################################SegNet的使用#######################################
    def segnet_infer(self,bgr_image):
        """
        通过segnet,获取lable_image
        @param bgr_image:bgr图片
        @return:label_image:640*480*分割种类
        """
        image=cv.resize(bgr_image,(640,480))#变换成640,480才可以送入网络中
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image=np.transpose(image,(2,0,1))
        image=self.norm_seg(torch.from_numpy(image.astype(np.float32)))
        image=torch.unsqueeze(image,0).cuda()
        label_image=self.segnet(image)#获取到的是一个1,60,480,640的Tensor
        label_image=label_image.cpu().detach().numpy()
        label_image=np.transpose(label_image[0],(1,2,0))
        label_image=cv.resize(label_image,(self.img_length,self.img_width))#最终resize成1280,720的尺寸
        return label_image

    def generate_kernel(self,x,y):
        return np.ones((x,y),np.uint8)

    def process_segnet_result(self,label_image,debug=False):
        """
        解析label_image,从一个
        @param label_image:
        @return:
        """
        #1:对每一层mask进行筛选,较少pixel的图层全部清零,认为是目标的图层做形态学操作提升图像轮廓品质
        for index in range(label_image.shape[2]):
            mask=label_image[:,:,index]
            sum_mask=np.sum(mask>0)


            if sum_mask<1500 or index==0:
                label_image[:,:,index]=0#如果这一层的pixel数目不够多,则全部变为0
            else:#认为是目标图片,先做形态学闭操作填充内部,再做形态学开操作,滤掉周围零星点
                label_image[:,:,index]=cv.morphologyEx(label_image[:,:,index],cv.MORPH_CLOSE,kernel=self.generate_kernel(20,20))
                label_image[:,:,index]=cv.morphologyEx(label_image[:,:,index],cv.MORPH_OPEN,kernel=self.generate_kernel(20,20))

            if debug:
                print("sum_mask is",sum_mask)
                if sum_mask>1500:
                    cv.imshow("mask",label_image[:,:,index])
                    cv.waitKey(0)

        #2:采用argmax得到最终的结果
        output_result=np.argmax(label_image,axis=2)

        return output_result

    def get_bgr_mask(self,bgr_image,debug=False):
        """
        送入bgr_image,得到语义分割的结果
        @param bgr_image:bgr图片
        @param debug: 是否进行debug(即可视化每一层的结果)
        @return:语义分割之后的结果,输出size和bgr_image相同,但是Channel=1,像素中,多少值即对应了多少的像素
        """
        label_image=self.segnet_infer(bgr_image)
        mask_result=self.process_segnet_result(label_image,debug=debug)
        return mask_result

    ###################################DenseFusion的使用#######################################
    def get_bbox_from_labelimage(self,label_image):
        """
        基于label获取物体的bbox
        @param label_image: SegNet分割得到的轮廓图
        @return:
        """
        rows = np.any(label_image, axis=1)
        cols = np.any(label_image, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        r_b = rmax - rmin
        for tt in range(len(self.border_list)):
            if self.border_list[tt] < r_b < self.border_list[tt + 1]:
                r_b = self.border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(self.border_list)):
            if self.border_list[tt] < c_b < self.border_list[tt + 1]:
                c_b = self.border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > self.img_width:
            delt = rmax - self.img_width
            rmax = self.img_width
            rmin -= delt
        if cmax > self.img_length:
            delt = cmax - self.img_length
            cmax = self.img_length
            cmin -= delt
        return rmin, rmax, cmin, cmax

    def get_objectlist_from_labelimage(self,label_image):
        """
        获取语义分割图片中存在的物体index
        @param label_image: 语义分割结果
        @return: object_list:包含了物体对应的id
        """
        object_list=[]
        temp_image=label_image.copy()
        while 1:
            max=np.max(temp_image)
            if max>0:
                object_list.append(max)
            temp_image[temp_image==max]=0
            if max==0:
                break
        return object_list

    def get_poses_withlabel(self,rgb_image,depth_image,label_image,object_list,debug=False):
        """
        送入rgb图片,深度图,Mask图,以及图片中的object_id,就可以得到所有物体的Pose
        @param rgb_image: rgb图片(而非bgr_image)
        @param depth_image: 深度图
        @param label_image: Mask图
        @param object_list: 图片中的对应索引
        @param debug: 是否进行debug,即是否展示结果
        @return:
        """
        save_result_list=[]#保存结果的list
        if debug:
            show_image=rgb_image.copy()

        for object_id in object_list:
            try:
                #1:获取深度图的iou
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth_image, 0))#提取出非0的depth_image
                mask_label = ma.getmaskarray(ma.masked_equal(label_image, object_id))#object_id对应的mask
                mask = mask_label * mask_depth  # 这个mask就是一个物体的轮廓
                rmin, rmax, cmin, cmax = self.get_bbox_from_labelimage(mask)

                if debug:
                    #绘制对应物体的bbox
                    cv.rectangle(show_image,(cmin,rmin),(cmax,rmax),(0,255,0),2)
                    cv.putText(show_image,"{}".format(object_id),(cmin,rmin),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                #2:选中图片需要索引的值
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                #2.1:如果choose超过选取的1000个点,则随机进行选取
                if len(choose) > self.num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:self.num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                #2.2:否则进行pad操作
                else:
                    choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                #3:基于choose,得到对应的深度点和RGB点
                #3.1:深度图点
                depth_masked = depth_image[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked =self. xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])
                pt2 = depth_masked / self.cam_scale
                pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)#得到的点云
                cloud = torch.from_numpy(cloud.astype(np.float32))

                #3.2:RGB点
                img_masked = np.array(rgb_image)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))

                #3.3:获取点云的索引点
                index = torch.LongTensor([object_id - 1])#这个用于选取点云参数

                #4:将值送入CUDA中
                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()
                cloud = cloud.view(1, self.num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                #5:通过DenseFusion生成预测RT
                pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)  # 得到了预测的R,T矩阵,以及对应的Center
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1,self.num_points, 1)
                pred_c = pred_c.view(self.bs, self.num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
                points = cloud.view(self.bs * self.num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

                save_result_list.append({'object_id':object_id,'rot':my_r,'trans':my_t})
            except:
                print("[warning],DenseFusion Detect Failed")

        if debug:
            show_points=[]
            axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            show_points.append(axis_point)
            cv.imshow("show_image",show_image)
            for result in save_result_list:
                object_id=result['object_id']
                rot=result['rot']
                trans=result['trans']
                Pose_Matrix=quaternion_matrix(rot)
                Pose_Matrix[0:3,3]=trans.T
                origin_cloud=o3d.geometry.PointCloud()
                origin_cloud.points=o3d.utility.Vector3dVector(self.object_points[object_id])
                origin_cloud.transform(Pose_Matrix)
                axis_point=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                axis_point.transform(Pose_Matrix)

                show_points.append(origin_cloud)
                show_points.append(axis_point)
            cv.waitKey(0)
            o3d.visualization.draw_geometries(show_points)

        print("DenseFuison Detect Success")
        return save_result_list

    def get_pose(self,bgr_image,depth_image,debug=False):
        """
        送入bgr图片和深度图,返回视野中所有物体在摄像头坐标系下的Pose
        @param bgr_image: bgr图片
        @param depth_image: 深度图
        @return: pose_results:一个list,每个元素为一个dict,dict中包含'object_id','rot','trans'
        """
        #1:首选先产生labelimage
        label_image=self.get_bgr_mask(bgr_image,debug=debug)

        #2:产生所有位置
        object_list=self.get_objectlist_from_labelimage(label_image)
        rgb_image=cv.cvtColor(bgr_image,cv.COLOR_BGR2RGB)
        pose_results=self.get_poses_withlabel(rgb_image,depth_image,label_image,object_list,debug=debug)
        return pose_results

    def see_detect_result(self,debug=False):
        """
        这里面用于看这个场景的识别结果
        @return:
        """
        read_Data=Read_Data.Read_Data(read_images=True)
        rate=rospy.Rate(10)
        with torch.no_grad():
            while not rospy.is_shutdown():
                if read_Data.bgr_image is not None and read_Data.depth_image is not None:
                    pose_result=self.get_pose(read_Data.bgr_image,read_Data.depth_image,debug=debug)
                    print("The result is:\n {}".format(pose_result))
                    break
                else:
                    print('[Warning] color_image not exist')
                rate.sleep()

    ###################################采用Service进行发布#######################################
    def get_worldframe_pose(self,pose_results):
        """
        将pose_results中的pose变换到世界坐标系中
        @param pose_results: 摄像头坐标系下的物体
        @return:
        """
        poseinworld_results=[]
        for pose_result in pose_results:
            object_id=pose_result['object_id']
            rot=pose_result['rot']
            trans=pose_result['trans']
            Pose_inCamera=quaternion_matrix(rot)
            Pose_inCamera[0:3,3]=trans.T
            Pose_inWorld=self.Trans_camera2world.dot(Pose_inCamera)
            world_rot=quaternion_from_matrix(Pose_inWorld)
            world_trans=Pose_inWorld[0:3,3].T
            poseinworld_results.append({'object_id':object_id,'rot':world_rot,'trans':world_trans})
        return poseinworld_results

    def get_pubinfo(self,pose_results):
        """
        从一个list变换为两个dict,分别存放id和姿态
        @param pose_results:
        @return:
        """
        object_id_list=[]
        poses_list=[]
        for pose_result in pose_results:
            object_id=pose_result['object_id']
            rot=pose_result['rot']
            trans=pose_result['trans']
            object_id_list.append(self.object_names[object_id])
            pose=Pose()
            pose.position.x=trans[0]
            pose.position.y=trans[1]
            pose.position.z=trans[2]
            ####DenseFusion输出的rot的顺序是wxyz,但是ros中默认的顺序是xyzw######
            # pose.orientation.x=rot[0]
            # pose.orientation.y=rot[1]
            # pose.orientation.z=rot[2]
            # pose.orientation.w=rot[3]
            pose.orientation.x=rot[1]
            pose.orientation.y=rot[2]
            pose.orientation.z=rot[3]
            pose.orientation.w=rot[0]
            poses_list.append(pose)
        return object_id_list,poses_list

    def pub_pose_array(self):
        """
        用于发布Pose的位置
        @return:
        """
        read_Data=Read_Data.Read_Data(read_images=True)
        rate=rospy.Rate(10)
        with torch.no_grad():
            while not rospy.is_shutdown():
                if read_Data.bgr_image is not None and read_Data.depth_image is not None:
                    pose_results=self.get_pose(read_Data.bgr_image,read_Data.depth_image)
                    poseinworld_result=self.get_worldframe_pose(pose_results)
                    objecd_id_list,poses_list=self.get_pubinfo(poseinworld_result)
                    modelStates=ModelStates()
                    modelStates.name=objecd_id_list
                    modelStates.pose=poses_list
                    self.poses_pub.publish(modelStates)
                else:
                    print('[Warning] color_image not exist')
                rate.sleep()
                if self.STOP_FLAG:
                    return

    def update_detect_state(self,data):
        state=data.data
        if state==1:
            print("DenseFusion Get Stop Flag,pub node will stop")
            self.STOP_FLAG=True

    ###################################进行DenseFusion精度确定#######################################
    def compare_pose(self,true_pose,pose):
        """
        用于比较两个pose的差距
        @param true_pose:
        @param pose:
        @return:
        """
        true_trans=np.array([true_pose.position.x,true_pose.position.y,true_pose.position.z])
        predict_trans=np.array([pose.position.x,pose.position.y,pose.position.z])
        dist=np.sqrt(np.sum(np.square(true_trans-predict_trans)))

        true_rot=np.array([true_pose.orientation.x,true_pose.orientation.y,true_pose.orientation.z,true_pose.orientation.w])
        predict_rot=np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])

        # true_rot=np.array([true_pose.orientation.w,true_pose.orientation.x,true_pose.orientation.y,true_pose.orientation.z])
        # predict_rot=np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])

        true_rot_rpy=np.array(trans_tools.euler_from_quaternion(true_rot))*180/math.pi
        predict_rot_rpy=np.array(trans_tools.euler_from_quaternion(predict_rot))*180/math.pi

        print("true_trans:  : {}".format(true_trans))
        print("preidct_trans: {}".format(predict_trans))
        print("true rot rpy : {}".format(true_rot))
        print("predict rotrpy:{}".format(predict_rot))

        rot_dist=np.arccos(np.abs(true_rot.dot(predict_rot.T)))

        # rot_dist=np.sqrt(np.sum(np.square(true_rot-predict_rot)))

        return dist,rot_dist

    def check_densefusion(self,scene_id="1-1",debug=True):
        """
        这里面用于确定DenseFusion的姿态识别和目标的姿态识别之间的差距
        @param scene_id: 场景id,默认为1-1
        @return:
        """
        read_Data=Read_Data.Read_Data(read_images=True)
        #获取Gazebo中读取的Pose
        world_info_list,gazebo_name2true_name,gazebo_name_list=read_Data.get_world_info(scene_id=scene_id)

        #获取DenseFusion中识别的Pose
        rate=rospy.Rate(10)
        densefusion_Detector=DenseFusion_Detector()
        with torch.no_grad():
            while not rospy.is_shutdown():
                if read_Data.bgr_image is not None and read_Data.depth_image is not None:
                    pose_result=self.get_pose(read_Data.bgr_image,read_Data.depth_image,debug=debug)
                    poseinworld_result=self.get_worldframe_pose(pose_result)
                    objecd_id_list,poses_list=self.get_pubinfo(poseinworld_result)
                    #比较之后得到结果就进行下一个场景,也跑一个auto_run进行解决
                    break
                else:
                    print('[Warning] color_image not exist')
                rate.sleep()

        #两个进行对比,从而知道结果
        #基于世界信息进行获取
        average_dist=0
        average_rot_dist=0
        for true_data in world_info_list:
            object_name=true_data['true_name']
            true_pose=true_data['model_pose']
            gazebo_name=true_data['gazebo_name']
            print("************ {}----{} pose***************".format(object_name,gazebo_name))
            try:
                index=objecd_id_list.index(object_name)
            except:
                print("[Warning] the SegNet detect wrong of"+object_name)
                continue

            pose=poses_list[index]
            dist,rot_dist=self.compare_pose(true_pose,pose)
            average_dist=average_dist+dist
            average_rot_dist=average_rot_dist+rot_dist
            # print("XYZ dist is",dist)
            # print("Rot dist is",rot_dist)

        # print("average_dist:",average_dist/len(world_info_list))
        # print("average_rot_dist",average_rot_dist/len(world_info_list))

if __name__ == '__main__':
    densefusion_Detector=DenseFusion_Detector(init_node=True)
    densefusion_Detector.pub_pose_array()#发布所有物体Pose
    # densefusion_Detector.check_densefusion("1-1")#确定场景识别精度
    # densefusion_Detector.see_detect_result(debug=True)#用于查看这个场景的识别结果
    # change_pth()#更改网络zip包


