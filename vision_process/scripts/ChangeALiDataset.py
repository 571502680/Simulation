## !/usr/bin/env python
## -*- coding: utf-8 -*-
"""
文件说明:
ALi数据集中,需要生成一套HSV外表的图片,从而可以生成分割图片
与此同时,还需要生成目标的points.xyz文件,从而进行神经网络识别


运行逻辑:
1.获取所有物体的名称
2.重新生成所有物体的颜色
3.获取所有场景中物体的尺寸
4.读取obj文件生成对应的xyz文件
5.生成只包含points.xyz的文件

更改记录:
Log:2020.9.8:
    这里面指定一个models路径,然后更改里面所有物体的visual mesh.
    存在一些小细节不算完善,比如object_list中缺少了prism(这个是缺少了颜色外表,之后颜色识别的时候可能会出问题),以及object_list这里面直接写死
    另外整个程序最好基于Python3运行,Python3测试可以正常运行,Python2未知

Log:2020.9.10:
    加入了prism这个类,让他也占据一个HSV的值,之后遇到了识别问题再去单独处理这个,先不管那一部分
    加入了generate_xyz_file,根据obj文件生成points.xyz(且目标都是collision.obj)
    加入了add_scale函数,对color_index进行了添加,最后一行是scale,目前只解决1-,2-,3-三个场景的问题

log:2020.9.24:
    加入了新的物体,将所有物体从60个添加到了80个,进行了代码的重新书写
"""
import sys
import os
import cv2 as cv
import numpy as np
import math
import open3d as o3d
import xml.etree.ElementTree as ET
import shutil

python_path=os.path.dirname(__file__)

class Change_Dataset:
    def __init__(self,material_path=None):
        """
        这个类用于对阿里给定的数据集进行变更
        1:将Models中物体的外表变换为固定颜色用于获得语义信息
        2:读取每一个物体的obj文件,获取物体的点,生成对应的xyz文件
        3:获取所有场景的.world文件,获取他们对应的scale从而进行重建
        """
        #1:指定对应路径
        #ocrtoc_materials路径(主目录)
        if material_path is None:
            self.material_path="/home/elevenjiang/Desktop/ocrtoc_materials"
        else:
            self.material_path=material_path

        self.models_path=self.material_path+"/"+"models_change"#models路径
        self.scenes_path=self.material_path+"/"+"scenes"#scenes路径

        #从classes.txt中获取物体名称
        self.objects_names=self.get_objects_name()

        #生成颜色编码方式,H采用180度分为30个档(6为间隔),S有85,179,255,V默认为200
        self.HSV_list,self.BGR_list=self.generate_color_list()

    ####################################生成classes.txt####################################
    def generate_classes_txt(self):
        """
        读取models文件下的所有文件夹,生成对应的名称
        这里面去掉了table和.gitignore两个文件
        :return:
        """
        class_names=os.listdir(self.models_path)
        class_names.sort()

        file=open("classes.txt",'w')
        for class_name in class_names:
            if class_name=="table" or class_name==".gitignore":
                continue
            file.write(class_name+"\n")
        file.close()

    def get_objects_name(self):
        """
        这里面进行此目录中classes.txt的文件更新
        :return:
        """
        file=open("classes.txt",'r')
        class_names=file.readlines()
        objects_name=[]
        for class_name in class_names:
            class_name=class_name[:-1]#去掉\n
            objects_name.append(class_name)
        return objects_name

    ###################################更改物体颜色部分的代码######################################
    def generate_color_list(self,v_value=200):
        """
        生成hsv颜色list,对H和S两个通道进行赋值,V通道默认200
        目前有59个物体(prism需要注意他是没有颜色的,这里面随便赋值,生成Mask的时候再去解决这个问题)
        H为59/len(s_index)
        S有85,127和2553个
        V默认200
        先写死,之后再灵活写
        @return: HSV_list,RGB_list
        """
        HSV_list=[]
        BGR_list=[]
        s_index=[85,170,255]
        v=v_value
        h_number=int(math.ceil(float(len(self.objects_names))/len(s_index)))#确保生成的h_number是超过object_names的
        h_range=int(180/h_number)
        for s in s_index:
            for i in range(h_number):
                HSV_list.append((int(i*h_range),s,v))
                b,g,r=self.hsv2bgr(int(i*h_range),s,v)
                BGR_list.append((b,g,r))
        return HSV_list,BGR_list

    def test_convert_rgbhsv(self,bgr=None,hsv=None):
        """
        测试rgb和hsv之间转换是否正确,给定bgr或者hsv,就会转换到另外一个
        @param bgr:
        @param hsv:
        @return:
        """
        if bgr is None:
            a1,a2,a3=hsv
            print("Your input is hsv:")
            print("\t H:{} S:{} V{}:".format(a1,a2,a3))
            print("Target output is :")
            o1,o2,o3=self.hsv2bgr(a1,a2,a3)#测试函数
            print("\t B:{} G:{} R{}:".format(o1,o2,o3))

        elif hsv is None:
            a1,a2,a3=bgr
            print("Your input is bgr:")
            print("\t B:{} G:{} R{}:".format(a1,a2,a3))
            print("Target output is :")
            o1,o2,o3=self.bgr2hsv(a1,a2,a3)#测试函数
            print("\t H:{} S:{} V{}:".format(o1,o2,o3))

        else:
            print("Please input rgb or hsv")
            sys.exit()

        image=np.ones((10,10),dtype=np.uint8)
        temp_image=np.array([image*a1,image*a2,image*a3])
        temp_image=np.transpose(temp_image,(1,2,0))
        temp_image=temp_image.astype(np.uint8)
        cv.imshow("input_image",temp_image)

        if bgr is not None:
            hsv_image=cv.cvtColor(temp_image,cv.COLOR_BGR2HSV)
            cv.imshow("hsv_image",hsv_image)
        else:
            bgr_image=cv.cvtColor(temp_image,cv.COLOR_HSV2BGR)
            cv.imshow("bgr_image",bgr_image)

        cv.waitKey(0)

    def bgr2hsv(self,b,g,r):
        """
        送入bgr值,得到对饮的hsv值
        @param r:
        @param g:
        @param b:
        @return:
        """
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        m = mx-mn
        h = 0#为了不报warning写的
        if mx == mn:
            h = 0
        elif mx == r:
            if g >= b:
                h = ((g-b)/m)*60
            else:
                h = ((g-b)/m)*60 + 360
        elif mx == g:
            h = ((b-r)/m)*60 + 120
        elif mx == b:
            h = ((r-g)/m)*60 + 240

        if mx == 0:
            s = 0
        else:
            s = m/mx
        v = mx
        H = int(h / 2)
        S = int(s * 255.0)
        V = int(v * 255.0)
        return H, S, V

    def hsv2bgr(self,h, s, v):
        """
        直接暴力用cvtColor完成任务算了
        @param h:
        @param s:
        @param v:
        @return:
        """
        image=np.ones((100,100),dtype=np.uint8)
        temp_image=np.array([image*h,image*s,image*v])
        temp_image=np.transpose(temp_image,(1,2,0))
        temp_image=temp_image.astype(np.uint8)

        output=cv.cvtColor(temp_image,cv.COLOR_HSV2BGR)
        b,g,r=output[0][0]

        return b,g,r

    def change_color(self,debug=False):
        """
        指定路径之后,可以改变所有物体的颜色,得到对应目标
        prism中没有对应的png,不进行对应图片的生成
        @param debug: debug为True时查看生成效果,False时是更改图片
        @return:
        """
        low=np.array([1])
        high=np.array([254])
        index_data=[]

        #1.读取每一个文件,并生成对应的参数
        for i,object_name in enumerate(self.objects_names):
            print("Now is changeing object: {}".format(object_name))
            if object_name=="table":
                print("[Warning] table should not appear in object_names")
                continue

            if object_name=='prism':#prism没有外观,但是仍然给他进行一个赋值,之后HSV的时候再说这个的解决办法
                b,g,r=self.BGR_list[i]
                h,s,v=self.HSV_list[i]
                index_data.append({'object_name':object_name,'color':(b,g,r),'hsv':(h,s,v)})
                continue

            #1.1:读取png
            visual_meshes_dir=os.path.join(self.models_path,object_name)+"/visual_meshes"
            temp_name=os.listdir(visual_meshes_dir)
            picture_dir=None
            #有图片不是textmap.png,因此采用这种方法
            for name in temp_name:
                if name[:-4]!=".dae":
                    picture_dir=visual_meshes_dir+"/"+name
                    break

            if picture_dir is None:
                print("[Error] The objet",object_name,"has problem,please check")
                sys.exit()
            testure_map=cv.imread(picture_dir)

            #1.2:生成目标图片
            #生成mask
            gray_image=cv.cvtColor(testure_map,cv.COLOR_BGR2GRAY)
            mask=cv.inRange(gray_image,low,high)

            #针对有黑色字的做一下形态学操作
            if object_name=="jenga" or object_name=="wood_block":
                mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel=np.ones((200,400),np.uint8))

            #生成对应颜色png
            b,g,r=self.BGR_list[i]
            h,s,v=self.HSV_list[i]
            
            h_mask=np.ones(gray_image.shape,dtype=np.float32)*h*mask/255
            s_mask=np.ones(gray_image.shape,dtype=np.float32)*s*mask/255
            v_mask=np.ones(gray_image.shape,dtype=np.float32)*v*mask/255

            #生成输出图片
            hsv_image=np.array([h_mask,s_mask,v_mask])
            hsv_image=np.transpose(hsv_image,(1,2,0))
            hsv_image=hsv_image.astype(np.uint8)
            rgb_image=cv.cvtColor(hsv_image,cv.COLOR_HSV2BGR)

            #保存图片
            if debug:
                print("Object Nmae:",object_name)
                print("HSV:",h,s,v)
                print("BGR:",b,g,r)
                cv.namedWindow("origin_image",cv.WINDOW_NORMAL)
                cv.imshow("origin_image",testure_map)
                cv.namedWindow("rgb_image",cv.WINDOW_NORMAL)
                cv.imshow("rgb_image",rgb_image)
                hsv_image=cv.cvtColor(rgb_image,cv.COLOR_BGR2HSV)#这里重新check一下HSV是正确的
                cv.namedWindow("hsv_image",cv.WINDOW_NORMAL)
                cv.imshow("hsv_image",hsv_image)
                cv.waitKey(0)
            else:
                cv.imwrite(picture_dir,rgb_image)

            index_data.append({'object_name':object_name,'color':(b,g,r),'hsv':(h,s,v)})


        #2:生成对应关系表
        color_index=open("color_index.txt",'w')
        for data in index_data:
            object_name=data['object_name']
            b,g,r=data['color']
            h,s,v=data['hsv']
            write_thing="{}:{},{},{},{},{},{},\n".format(object_name,b,g,r,h,s,v)
            color_index.write(write_thing)
        color_index.close()
        print("Already change the object color and generate the color_index.txt")

    ####################################查询所有scale部分####################################
    def add_scale(self):
        """
        在1-,2-,3-场景中,scale都是相同的,但是4,5两个场景中的物体却是不一定的,因此,目前只基于前3个场景进行尺寸生成
        所有的index变换成csv文件,从而可以更好的进行索引
        @return:
        """
        #生成一个空的dict用于scale的存储
        scale_dict={}
        for true_name in self.objects_names:
            scale_dict[true_name]=[]

        #寻找所有的scene文件
        files=os.listdir(self.scenes_path)
        for file in files:
            if file[0]=='4' or file[0]=='5':
                continue
            world_file=os.path.join(self.scenes_path,file)+"/input.world"
            xml_root=ET.parse(world_file).getroot()
            for child_1 in xml_root.findall('world/model'):
                gazebo_name=child_1.attrib['name']
                if gazebo_name == "ground_plane" or gazebo_name == "table":
                    continue

                #1:读取物体真实名称
                true_name=None
                for child_2 in child_1.findall('link/collision/geometry/mesh/uri'):
                    uri=child_2.text
                    uri=uri[8:]
                    stop=uri.find('/')
                    true_name=uri[:stop]
                    # if true_name=="prism":
                    #     #这里面更新一下prism的东西
                    #     pass

                if true_name is None:
                    print("[Warning] {} file {} don't have true_name".format(world_file,gazebo_name))

                #2:获取物体scale
                scale=None
                for child_2 in child_1.findall('link/collision/geometry/mesh/scale'):
                    scale_info=child_2.text
                    scale_info=scale_info.split(' ')
                    if scale_info[0]!=scale_info[1]:
                        print("[Warning] {} scale_info rate is not all same".format(world_file))
                    scale=float(scale_info[0])
                    break
                if scale is None:
                    print("[Warning] {} file {} don't have scale".format(world_file,scale))

                #3:检索已经存在的scale,看看是否会不同
                if scale in scale_dict[true_name]:
                    continue
                else:
                    if len(scale_dict[true_name])==0:
                        scale_dict[true_name].append(scale)
                    else:
                        print("Find {} object has differennt scale,it is in :{} ".format(true_name,world_file))
                        print("The origin is {}".format(scale_dict[true_name]))
                        print("The new is ".format(scale))

                        scale_dict[true_name].append(scale)

        #输出scale_dict,查看是否进行保存
        for object in scale_dict:
            print("object name is:{},its scale is:{}".format(object,scale_dict[object]))
            if len(scale_dict[object])>1:
                print("!!!!!!!This Object Scale is not only one!!!!!!!")


        temp=raw_input("Generate all_index.txt?\n          ---y for yes,else for no")
        print temp
        if temp !='y':
            return

        print("Begin to Generate all_index.txt...")
        #4:最后在color_index的txt中进行更多的添加
        color_index=open("color_index.txt",'r')
        all_index=open('all_index.txt','w')
        data=color_index.readlines()

        #4.1:对于每个物体进行处理,保存他们对应的scale
        for line in data:
            name=line.split(':')[0]
            if len(scale_dict[name])==0:
                save_scale=1.0
            else:
                save_scale=scale_dict[name][0]
            save_info=line[:-1]+"{},\n".format(save_scale)
            print("all_index file has write:  {}".format(save_info))
            all_index.write(save_info)

    ####################################生成xyz点的代码####################################
    def generate_xyz_file(self):
        """
        读取每个文件的.obj文件,然后在同目录下生成一个对应的xyz文件
        @return:
        """
        #1:读取所有物体的scale(仅适用于1-3场景)
        scale_list={}
        all_index=open("all_index.txt")
        for line in all_index:
            name=line.split(':')[0]
            number=line.split(':')[1]
            number=number.split(",")
            scale=float(number[-2])
            scale_list[name]=scale

        #1:索引所有文件
        for i,object_name in enumerate(self.objects_names):
            if  object_name=="table":
                continue
            #1.1:读取obj文件
            meshes_dir=os.path.join(self.models_path,object_name)+"/collision_meshes"
            temp_name=os.listdir(meshes_dir)
            obj_name=None
            for name in temp_name:
                if name[-4:]!=".obj":
                    continue
                else:
                    obj_name=name
                    break

            if obj_name is None:
                print("[Error] Can not find the obj file in {}".format(obj_name))
                sys.exit()

            print("********************************")
            print("now the {} object".format(i))
            obj_dir=meshes_dir+"/"+obj_name
            tri_mesh=o3d.io.read_triangle_mesh(obj_dir)
            point_cloud=tri_mesh.sample_points_poisson_disk(3500)
            np_points=np.asarray(point_cloud.points)
            np_points=np_points*scale_list[object_name]
            print("this object scale is :{}".format(scale_list[object_name]))

            #1.2:生成xyz文件
            xyz_path=meshes_dir+"/points.xyz"
            file=open(xyz_path,'w')
            for point in np_points:
                file.write("{} {} {}\n".format(point[0],point[1],point[2]))

            print("already generate {}'s points.xyz".format(object_name))

    def generate_object_models(self):
        """
        用于将所有物体的.xyz文件移动到object_models/object_name/下面
        :return:
        """
        if not os.path.exists("object_models"):
            print("Not exist object_models path,create a new file")
            os.mkdir("object_models")


        for i,object_name in enumerate(self.objects_names):
            if object_name=="table":
                print("[Warning] table should not appear in object_names")
                continue
            print("Now is Moving object {} ...".format(object_name))
            xyz_file=os.path.join(self.models_path,object_name)+"/collision_meshes/points.xyz"
            object_path="object_models/"+object_name+"/points.xyz"

            #新建文件夹用于保存对应文件
            if not os.path.exists("object_models/"+object_name):
                print("[Warning] Not exist {} file,create a new file".format(object_name))
                os.mkdir("object_models/"+object_name)

            #确保存在.xyz文件存在
            if not os.path.exists(xyz_file):
                print("[Error] Can not open object {}'s xyz file,please check path:{}".format(object_name,xyz_file))
                sys.exit()

            shutil.copyfile(src=xyz_file,dst=object_path)

        print("Move Finish")



####################使用样例#######################
def example():
    change_Dataset=Change_Dataset()#定义更改数据集的类
    change_Dataset.generate_classes_txt()#生成所有类别txt的文件
    change_Dataset.change_color()#更改所有图片的颜色
    change_Dataset.add_scale()#索引所有场景的scale,从而生成统一的尺寸
    change_Dataset.generate_xyz_file()#生成points.xyz文件
    change_Dataset.generate_object_models()#将所有points.xyz文件移动到object_models中


def main():
    change_Dataset=Change_Dataset()
    change_Dataset.generate_object_models()

if __name__ == '__main__':
    main()