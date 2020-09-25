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

class Auto_MakeData:
    def __init__(self,HSV_MODE):
        """
        这里面不断地制作图片,然后生成目标路径
        """
        self.HSV_MODE=HSV_MODE
        self.make_data_index_file=open('Make_Data_index.txt','a')
        

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


if __name__ == '__main__':
    auto_MakeData=Auto_MakeData(HSV_MODE=True)
    auto_MakeData.auto_run(begin_id='1-570',stop_id='2-1')
    # auto_MakeData.clean_dataset()
    # auto_MakeData.run_error_data()



