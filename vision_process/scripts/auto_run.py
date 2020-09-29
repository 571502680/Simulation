#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
这里面进行数据集的自动生成工作
Log: 2020.9.14:
    今天进行命令的整合,sort中按照顺序排序,从而知道在哪里没有
    另外,还添加了数据集检查的部分,确保所有数据集正常生成,错误的数据集进行了记录
    深度图中,有一些深度图的返回值是.nan,对于这些深度图进行了特殊的处理

Log: 2020.9.24:
    添加了基于Sapien的数据集生成接口
"""
import os
import sys
import time
import datetime
import glob
import cv2 as cv
import numpy as np
import scipy.io as scio
import open3d as o3d

import Read_Data
import Make_Data



class Auto_MakeData:
    def __init__(self,HSV_MODE):
        """
        这里面不断地制作图片,然后生成目标路径
        """
        self.HSV_MODE=HSV_MODE
        self.make_data_index_file=open('Make_Data_index.txt','a')
        self.python_path=os.path.dirname(os.path.abspath(__file__))

    def add_txt_time(self):
        self.make_data_index_file=open('Make_Data_index.txt','a')
        now_time=datetime.datetime.now()
        self.make_data_index_file.write('\n')
        self.make_data_index_file.write('\n')
        self.make_data_index_file.write("*"*50+'\n')
        self.make_data_index_file.write(str(now_time))
        if self.HSV_MODE:
            self.make_data_index_file.write("HSV Mode: True\n")
        else:
            self.make_data_index_file.write("HSV Mode: False\n")
        self.make_data_index_file.write("*"*50+'\n')

        self.make_data_index_file.close()

    def auto_run(self,begin_id=None,stop_id=None):
        #1:进行make_data的场景写入
        self.add_txt_time()

        #1:获取所有场景id
        all_scenes=os.listdir("/root/ocrtoc_materials/scenes")
        all_scenes.sort(key=lambda data:int(data[0])*10000+int(data.split('-')[1]))#按照排列顺序进行

        #2:对所有场景进行操作
        for i,scene_id in enumerate(all_scenes):
            #不引入4,5的错误场景
            if scene_id[0]=='4' or scene_id[0]=='5':
                continue

            #输出新场景并写入文件中(每一次都开启关闭一次,从而可以更好地查看位置)
            self.make_data_index_file=open('Make_Data_index.txt','a')
            print("!!!!!!!!!!!!!!!!!!!New Scene!!!!!!!!!!!!!!!!!!!!:{}".format(scene_id))
            self.make_data_index_file.write(scene_id+'\n')
            self.make_data_index_file.close()

            #2.1:指定开始和结束id
            if begin_id is not None:
                if scene_id!=begin_id:
                    continue
                else:
                    begin_id=None
            if stop_id is not None:#当出现停止id的时候,程序退出
                if scene_id==stop_id:
                    print("Arrive the Sotp,break")
                    sys.exit()

            #2.2:一定执行时间之后让其开始休息
            if i%20==0 and i>10:
                print("#"*50+"Now it run {} data".format(i)+"!"*50)
                print("Now it sleep for 20 sec")
                time.sleep(20)

            #2.3:进行任务执行
            if self.HSV_MODE:
                os.system("roslaunch vision_process make_data.launch scene:={} simulator:={}&".format(scene_id,"gazebo"))
            else:
                os.system("roslaunch vision_process make_data.launch scene:={} simulator:={}&".format(scene_id,"sapien"))
            time.sleep(15)

            os.system("rosrun vision_process Make_Data.py {} {}".format(scene_id,self.HSV_MODE))
            time.sleep(1)

            if self.HSV_MODE:
                #关闭掉Gazebo的内容
                os.system("killall gzserver")
            else:
                #关闭sapien的内容
                os.system("killall sapien_env.py")
                os.system("killall roslaunch")
                os.system("killall python2")
                os.system("killall robot_state_pub")
                time.sleep(1)

            os.system("rosclean purge -y")#清除ros内存
        print("!!!!!!!!!!!!!!!!!!!Already Make All Data!!!!!!!!!!!!!!!!!!!")

    def run_error_data(self):
        """
        读取error_data.txt,对里面所有的scene_id进行重新数据生成
        @return:
        """
        #1:进行make_data的场景写入
        self.add_txt_time()

        #1:获取所有场景id
        error_data=open("error_data.txt",'r')
        all_scenes=error_data.readlines()

        #2:对所有场景进行操作
        for i,scene_id in enumerate(all_scenes):
            scene_id=scene_id[:-1]

            #不引入4,5的错误场景
            if scene_id[0]=='4' or scene_id[0]=='5':
                continue

            #2.2:一定执行时间之后让其开始休息
            if i%20==0 and i>10:
                print("#"*50+"Now it run {} data".format(i)+"!"*50)
                print("Now it sleep for 20 sec")
                time.sleep(20)

            #2.3:进行任务执行
            if self.HSV_MODE:
                os.system("roslaunch vision_process make_data.launch scene:={} simulator={}&".format(scene_id,"gazebo"))
            else:
                os.system("roslaunch vision_process make_data.launch scene:={} simulator={}&".format(scene_id,"sapien"))

            time.sleep(10)
            os.system("rosrun vision_process Make_Data.py {} {}".format(scene_id,self.HSV_MODE))
            time.sleep(1)
            os.system("killall gzserver")
            os.system("rosclean purge -y")

        print("!!!!!!!!!!!!!!!!!!!Already Make All Data!!!!!!!!!!!!!!!!!!!")

    def clean_dataset(self):
        """
        用于清理数据,同时输出存在问题的数据
        @return:
        """
        #1:获取所有场景id
        all_scenes=os.listdir("/root/ocrtoc_materials/scenes")
        all_scenes.sort(key=lambda data:int(data[0])*10000+int(data.split('-')[1]))#按照排列顺序进行

        error_data_file=open("error_data.txt",'w')
        dataset_dir="../ALi_dataset/Dataset/"
        find_file=['-color.png','-depth.png','-label.png','-meta.mat']

        #2:对所有场景进行操作
        for i,scene_id in enumerate(all_scenes):
            #不引入4,5的错误场景
            if scene_id[0]=='4' or scene_id[0]=='5':
                continue

            #查看这个场景中是否4个文件都包含,不包含则删除,并把场景id进行记录
            for end_name in find_file:
                if not os.path.exists(dataset_dir+scene_id+end_name):
                    error_data_file.write(scene_id+"\n")
                    #删除掉所有这一种后缀的文件
                    all_scenes_file=glob.glob(dataset_dir+scene_id+'-*')
                    for file in all_scenes_file:
                        os.remove(file)
                    break

        error_data_file.close()

    def makedata_onescene(self,scene_id):
        """
        对scene_id做数据生成
        @param scene_id:
        @return:
        """
        os.system("roslaunch vision_process make_data.launch scene:={} &".format(scene_id))
        time.sleep(10)
        os.system("rosrun vision_process Make_Data.py {} {}".format(scene_id,self.HSV_MODE))
        time.sleep(1)
        os.system("killall gzserver")
        os.system("rosclean purge -y")

    def compare_pose(self):
        """
        这里面进行label和rgb的比较,从而知道轮廓是否正确
        :return:
        """
        #1:获取所有场景id:
        all_scenes=glob.glob(self.python_path+"/ALi_Dataset/data/*-color.png")
        all_scenes_id=[]
        for scene in all_scenes:
            file_name=scene.split('data/')[1]
            index=file_name.rfind('-')
            scene_id=file_name[:index]
            all_scenes_id.append(scene_id)
        all_scenes_id.sort(key=lambda data:int(data[0])*10000+int(data.split('-')[1]))#按照排列顺序进行

        all_i=240
        while all_i<len(all_scenes_id):
            scene_id=all_scenes_id[all_i]
            image=cv.imread(self.python_path+"/ALi_Dataset/data/"+scene_id+"-color.png")
            label=cv.imread(self.python_path+"/ALi_Dataset/data/"+scene_id+"-label.png")

            #label图片变换成3通道的
            cv_image=label.copy()
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)

            #两个图片融合起来
            merge_image=cv.addWeighted(image,0.5,color_map,0.5,0)
            cv.imshow("merge_image",merge_image)
            input_temp=cv.waitKey()
            print("Now the Scene id is: {}".format(scene_id))
            if input_temp==100:#'d'
                print("You input d to delete the scene:{}".format(scene_id))

            if input_temp==115:#'s'
                print("You input s,Stop")
                return

            if input_temp==98:#'b'
                print("You input b,See before image")
                all_i=all_i-2

            all_i=all_i+1

    def check_pose(self):
        """
        这里面是确定meta的pose是否生成正确
        :return:
        """
        all_scenes=glob.glob(self.python_path+"/ALi_Dataset/data/*-color.png")
        all_scenes_id=[]
        for scene in all_scenes:
            file_name=scene.split('data/')[1]
            index=file_name.rfind('-')
            scene_id=file_name[:index]
            all_scenes_id.append(scene_id)
        all_scenes_id.sort(key=lambda data:int(data[0])*10000+int(data.split('-')[1]))#按照排列顺序进行

        for scene_id in all_scenes_id:
            #开启对应的sapien
            os.system("roslaunch ocrtoc_task bringup_simulator.launch scenes:={} &".format(scene_id))
            time.sleep(5)
            print("*"*50)
            make_Data=Make_Data.Make_Data(HSV_mode=True)
            read_YCB=Read_Data.Read_YCB(get_object_points=True)
            read_Data=Read_Data.Read_Data(simulator='sapien')

            meta = scio.loadmat(make_Data.dataset_pth+'/{}-meta.mat'.format(scene_id))


            poses=meta['poses']
            print(poses)
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
    auto_MakeData=Auto_MakeData(HSV_MODE=False)
    auto_MakeData.auto_run(begin_id='2-78')
    # auto_MakeData.clean_dataset()
    # auto_MakeData.run_error_data()
    # auto_MakeData.compare_pose()
    # auto_MakeData.check_pose()



