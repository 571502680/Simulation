#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
这里封装DenseFusion的使用函数,从而让整体可以前向推理
Log:2020.9.14:
    开始进行DenseFusion封装
"""

import argparse
import copy
import sys

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

sys.path.append("/root/ocrtoc_ws/src/vision_process/scripts")
sys.path.append("/root/ocrtoc_ws/src/vision_process/scripts/DenseFusion_Lib")
sys.path.append("/root/ocrtoc_ws/src/vision_process/scripts/SegNet_Lib")


import Make_Data
#DenseFusion库
from DenseFusion_Lib.network import PoseNet,PoseRefineNet
from DenseFusion_Lib.transformations import quaternion_matrix,quaternion_from_matrix,euler_from_quaternion

#SegNet库
from SegNet_Lib.segnet import SegNet

class DenseFusion_Detector:
    def __init__(self,model_path=None,refine_model_path=None,segnet_path=None):
        #1:导入网络
        if model_path is None:
            self.model_path="DenseFusion_Lib/models/pose_model_1_0.04200344247743487.pth"
        else:
            self.model_path=model_path

        if refine_model_path is None:
            self.refine_model_path="models/temp.pth"
        else:
            self.refine_model_path=refine_model_path

        self.num_points=1000
        self.num_obj=59
        self.estimator = PoseNet(num_points=self.num_points, num_obj=self.num_obj)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(self.model_path))
        self.estimator.eval()

        # refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
        # refiner.cuda()
        # refiner.load_state_dict(torch.load(opt.refine_model))
        # refiner.eval()


        #2:一系列需要的参数
        #扩宽参数
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
        class_file = open('classes.txt')  # 读取一共包含的类别信息
        class_id = 1
        # self.object_points = {}
        # while 1:
        #     class_input = class_file.readline()
        #     if not class_input:
        #         break
        #     class_input = class_input[:-1]  # 应该用于去掉换行符号
        #     # 得到对应的model
        #     input_file = open('object_models/{0}/points.xyz'.format(class_input))  # 得到他们对应的xyz文件
        #     self.object_points[class_id] = []
        #     while 1:
        #         input_line = input_file.readline()
        #         if not input_line:
        #             break
        #         input_line = input_line[:-1]
        #         input_line = input_line.split(' ')
        #         self.object_points[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])  # ????????xyz????
        #     input_file.close()
        #     self.object_points[class_id] = np.array(self.object_points[class_id])  # 将一个list变换为一个np.array
        #     class_id += 1

        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_input = class_input[:-1]  # 应该用于去掉换行符号
            # 得到对应的model
            input_file = open('object_models/{0}/points.xyz'.format(class_input))  # 得到他们对应的xyz文件
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1]
                input_line = input_line.split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])  # ????????xyz????
            input_file.close()
            self.cld[class_id] = np.array(self.cld[class_id])  # 将一个list变换为一个np.array
            class_id += 1





        #4:导入SegNet
        self.segnet=SegNet(input_nbr=3,label_nbr=60)
        if segnet_path is None:
            self.segnet_path="SegNet_Lib/models/model_47_0.01935524859049536.pth"
        else:
            self.segnet_path=segnet_path
        self.segnet.load_state_dict(torch.load(self.segnet_path))
        self.segnet=self.segnet.cuda()
        self.norm_seg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def get_bbox_from_label(self,label):
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
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

    def get_label_image(self,bgr_image):
        """
        通过segnet,获取bgr_image
        @param bgr_image:
        @return:
        """
        image=cv.resize(bgr_image,(640,480))
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image=np.transpose(image,(2,0,1))
        image=self.norm_seg(torch.from_numpy(image.astype(np.float32)))
        image=torch.unsqueeze(image,0).cuda()
        label_image=self.segnet(image)#获取到的是一个1,60,480,640的Tensor


        label_image=label_image.cpu().detach().numpy()
        label_image=np.transpose(label_image[0],(1,2,0))

        return label_image

    def see_result_segnet(self,label_image):
        """
        这里面对结果进行可视化,从而知道哪个是最可能的结果
        这里面可以出来,但是非常的不优雅,还是想一个好办法让他可以直接输出对应结果的值是最好的.目前为了尽快完成任务就显不处理这里
        @param label_image:
        @return:
        """
        output_result=np.zeros((720,1280),dtype=np.uint8)
        for index in range(label_image.shape[2]):
            mask=label_image[:,:,index]
            mask=cv.resize(mask,(1280,720))
            sum_mask=np.sum(mask>0)
            # print("index is {},sum_mask is{}".format(index,sum_mask))
            if sum_mask<4000 or index==0:
                continue
            else:

                output_result[mask>0]=index

        return output_result

    def get_poses_fromlabel(self,rgb_image,depth_image,label_image,object_list,debug=False):
        """
        送入rgb图片,深度图,Mask图,以及图片中的object_id,就可以得到所有物体的Pose
        @param rgb_image: rgb图片
        @param depth_image: 深度图
        @param label_image: Mask图
        @param object_list: 图片中的对应索引
        @param debug: 是否进行debug,即是否展示结果
        @return:
        """
        save_result_list=[]
        if debug:
            show_image=rgb_image.copy()

        for object_id in object_list:
            try:
                #获取深度图的iou
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth_image, 0))#提取出非0的depth_image
                mask_label = ma.getmaskarray(ma.masked_equal(label_image, object_id))#object_id对应的mask
                mask = mask_label * mask_depth  # 这个mask就是一个物体的轮廓


                rmin, rmax, cmin, cmax = self.get_bbox_from_label(mask)

                #绘制对应物体的bbox
                if debug:
                    cv.rectangle(show_image,(cmin,rmin),(cmax,rmax),(0,255,0),2)
                    cv.putText(show_image,"{}".format(object_id),(cmin,rmin),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                #选中mask中的参数
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                #如果choose超过选取的1000个点,则随机进行选取
                if len(choose) > self.num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:self.num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                #否则进行pad操作
                else:
                    choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                #基于choose,得到对应的深度点和RGB点
                depth_masked = depth_image[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked =self. xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])
                pt2 = depth_masked / self.cam_scale
                pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)#得到的点云
                cloud = torch.from_numpy(cloud.astype(np.float32))
                img_masked = np.array(rgb_image)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]


                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([object_id - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()


                cloud = cloud.view(1, self.num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                # 此处开始预测任务,得到轮廓之后进行处理
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
                print("[warning]")

        if debug:
            show_points=[]
            cv.imshow("show_image",show_image)
            cv.waitKey(0)
            for result in save_result_list:
                object_id=result['object_id']
                rot=result['rot']
                trans=result['trans']
                Pose_Matrix=quaternion_matrix(rot)
                Pose_Matrix[0:3,3]=trans.T
                origin_cloud=o3d.geometry.PointCloud()
                origin_cloud.points=o3d.utility.Vector3dVector(self.cld[object_id])

                origin_cloud.transform(Pose_Matrix)

                show_points.append(origin_cloud)

            o3d.visualization.draw_geometries(show_points)


        return save_result_list

    def get_objectlist_from_label(self,label_image):
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


    def get_pose(self,bgr_image,depth_image):
        #1:首选先产生labelimage
        label_image=self.see_result_segnet(self.get_label_image(bgr_image))

        cv.imshow("label_image",label_image)
        object_list=self.get_objectlist_from_label(label_image)

        rgb_image=cv.cvtColor(bgr_image,cv.COLOR_BGR2RGB)


        #2:产生所有位置
        result=self.get_poses_fromlabel(rgb_image,depth_image,label_image,object_list,debug=True)
        return result


if __name__ == '__main__':
    make_Data=Make_Data.Make_Data()
    rospy.init_node("DenseFusion")
    rate=rospy.Rate(10)
    densefusion_Detector=DenseFusion_Detector()
    with torch.no_grad():
        while not rospy.is_shutdown():
            make_Data.get_images()
            if make_Data.color_image is not None and make_Data.depth_image is not None:
                pose_result=densefusion_Detector.get_pose(make_Data.color_image,make_Data.depth_image)
                print("The result is:",pose_result)
                break
            else:
                print('[Warning] color_image not exist')
            rate.sleep()












