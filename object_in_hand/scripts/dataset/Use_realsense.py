"""
相机类,用于驱动Realsense
"""
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
class RS:
    """
    D435的类
    """
    def __init__(self,open_depth=True,open_color=True,frame=30,resolution='1280x720',align_to_depth=True,use_filter=True):
        self.open_depth=open_depth
        self.open_color=open_color

        #开启通信接口
        self.pipeline = rs.pipeline()

        #像素参数初始化,并定义内参矩阵,内参矩阵从rs-sensor-control中提取
        #x_matrix和y_matrix用于进行xy的计算
        all_matrix=np.load("all_matrix.npz")
        if resolution=='640x480':
            self.image_width=640
            self.image_height=480
            self.fx=383.436
            self.fy=383.436
            self.cx=318.613
            self.cy=238.601
            self.x_matrix=all_matrix['x_matrix640']
            self.y_maxtrix=all_matrix['y_matrix640']
        elif resolution=='1280x720':
            self.image_width=1280
            self.image_height=720
            self.fx=639.059
            self.fy=639.059
            self.cx=637.688
            self.cy=357.688
            self.x_matrix=all_matrix['x_matrix1280']
            self.y_maxtrix=all_matrix['y_matrix1280']
        elif resolution=='848x480':
            self.image_width=848
            self.image_height=480
            self.fx=423.377
            self.fy=423.377
            self.cx=422.468
            self.cy=238.455
            self.x_matrix=all_matrix['x_matrix848']
            self.y_maxtrix=all_matrix['y_matrix848']
        else:
            assert False,"请输入正确的resolution值"

        #设置像素操作
        config_rs = rs.config()
        if open_depth:
            config_rs.enable_stream(rs.stream.depth, self.image_width, self.image_height, rs.format.z16, frame)
            self.depth_image=None
            self.color_map=None

        if open_color:
            config_rs.enable_stream(rs.stream.color, self.image_width, self.image_height, rs.format.bgr8, frame)
            self.color_image=None

        #开始通信流
        self.profile=self.pipeline.start(config_rs)

        #当RGB和深度同时开启时,将颜色图向深度图对齐
        if align_to_depth:
            if open_depth and open_color:
                align_to=rs.stream.depth
                self.align=rs.align(align_to)

        else:
            if open_depth and open_color:
                align_to=rs.stream.color
                self.align=rs.align(align_to)

        #定义滤波器
        self.use_filter=use_filter
        self.dec_filter=rs.decimation_filter(4)#降采样
        # self.temp_filter=rs.temporal_filter(3)#上下帧之间利用时间信息避免跳动,参数看官方文档
        self.hole_filter=rs.hole_filling_filter(2)#hole填充

        #用于读取深度值
        self.read_image_dpeth=None


    def get_data(self):
        """
        获取图像,统一对齐深度图
        深度图部分加上了滤波的部分
        输出:返回color_image和depth_image
        """
        frames=self.pipeline.wait_for_frames()

        #同时开启,与深度图对齐
        if self.open_color and self.open_depth:
            #获取深度图并滤波
            aligned_frames = self.align.process(frames)
            depth_frame=aligned_frames.get_depth_frame()
            #不同滤波器使用
            if self.use_filter:
                hole_filtered=self.hole_filter.process(depth_frame)
                dec_filtered=self.dec_filter.process(hole_filtered)
                depth_image = np.asanyarray(dec_filtered.get_data())
                depth_image=cv.resize(depth_image,(self.image_width,self.image_height))
            else:
                depth_image=np.asanyarray(depth_frame.get_data())

            #获取颜色图
            aligned_color_frame=aligned_frames.get_color_frame()
            color_image= np.asanyarray(aligned_color_frame.get_data())


        #开始单一深度图,并进行滤波
        elif self.open_depth and not self.open_color:
            #获取深度图
            depth_frame=frames.get_depth_frame()
            #进行不同滤波器处理
            if self.use_filter:
                hole_filtered=self.hole_filter.process(depth_frame)
                dec_filtered=self.dec_filter.process(hole_filtered)
                depth_image = np.asanyarray(dec_filtered.get_data())
                depth_image=cv.resize(depth_image,(self.image_width,self.image_height))
            else:
                depth_image=np.asanyarray(depth_frame.get_data())

            color_image=None

        else:
            color_image=None
            depth_image=None

        self.color_image=color_image
        self.depth_image=depth_image

        return color_image,depth_image

    def get_color_map(self,depth_image=None,range=None):
        """
        送入深度图,返回对应的颜色图
        :param depth_image:需要生成的颜色图,如果为None,则选取自带的深度图
        :param range: 是否需要滤除掉一定距离之后的值
        :return:
        """
        if depth_image is None:
            depth_image=self.depth_image

        range_image=depth_image.copy()
        if range is not None:
            depth_mask=cv.inRange(depth_image,0,range)
            cv.imshow("depth_mask",depth_mask)
            range_image=depth_image*depth_mask/255

        #开始转深度图
        color_map=range_image.copy()
        cv.normalize(color_map,color_map,255,0,cv.NORM_MINMAX)
        color_map=color_map.astype(np.uint8)
        color_map=cv.applyColorMap(color_map,cv.COLORMAP_JET)
        self.color_map=color_map

        return color_map

    def get_xyz_image(self):
        """
        基于深度图,获取一张xyz_image的图,3通道,分别存放了该像素点的xyz值
        :return:xyz_image
        """
        xyz_image=np.array([self.x_matrix*self.depth_image,self.y_maxtrix*self.depth_image,self.depth_image])
        xyz_image=xyz_image.transpose((1,2,0))
        return xyz_image

    def get_xyz(self,point,range_area=2):
        """
        获取point点的xyz值
        当索引到边上时,会直接所以该点的Z值
        :param point:需要获取xyz的像素点
        :param range_area:取周围邻域的中间值
        :return:np.array((X,Y,Z))
        """
        u,v=point
        u=int(u)
        v=int(v)
        center_Z=[]
        #1:对center_Z进行排序,得到中值作为深度
        try:
            for x in range(-range_area,range_area+1):
                for y in range(-range_area,range_area+1):
                    center_Z.append(self.depth_image[v-y,u-x])#采用行列索引
            center_Z.sort()
            Z=center_Z[int(len(center_Z)/2)]
        except:
            try:
                Z=self.depth_image[v,u]
            except:
                Z=0

        #2:使用外参进行反解
        X=(u-self.cx)*Z/self.fx
        Y=(v-self.cy)*Z/self.fy
        return np.array((X,Y,Z))

    ###***功能性函数**###
    def generate_xy_matrix(self):
        """
        用于生成xyz_image的矩阵
        测距点本质上至于z有关,向平面的xy是固定的,有z进行放大缩小
        会在这个的目录下生成all_matrix.npz文件,其中包含了对应需要的参数矩阵

        #使用方法:
        # data=np.load("all_matrix.npz")
        # x_640=data['x_matrix640']
        # x_1280=data['x_matrix1280']
        # print(x_640.shape)
        # print(x_1280.shape)
        :return:
        """

        #1:生成1280的矩阵
        x_1280_matrix=np.zeros((720,1280))
        y_1280_matrix=np.zeros((720,1280))
        fx=639.059
        fy=639.059
        cx=637.688
        cy=357.688
        for i in tqdm(range(1280)):
            for j in range(720):
                # print(temp_1280[j,i])#默认的索引是行列索引
                x_1280_matrix[j,i]=(i-cx)/fx
                y_1280_matrix[j,i]=(j-cy)/fy



        #2:生成640的矩阵
        x_640_matrix=np.zeros((480,640))
        y_640_matrix=np.zeros((480,640))
        fx=383.436
        fy=383.436
        cx=318.613
        cy=238.601
        for i in tqdm(range(640)):
            for j in range(480):
                x_640_matrix[j,i]=(i-cx)/fx
                y_640_matrix[j,i]=(j-cy)/fy



        #3:生成848x480的内参
        x_848_matrix=np.zeros((480,848))
        y_848_matrix=np.zeros((480,848))
        fx=423.377
        fy=423.377
        cx=422.468
        cy=238.455

        for i in tqdm(range(848)):
            for j in range(480):
                x_848_matrix[j,i]=(i-cx)/fx
                y_848_matrix[j,i]=(j-cy)/fy

        #保存对应的矩阵
        np.savez('all_matrix.npz',x_matrix640=x_640_matrix,y_matrix640=y_640_matrix,x_matrix1280=x_1280_matrix,y_matrix1280=y_1280_matrix,x_matrix848=x_848_matrix,y_matrix848=y_848_matrix)

    def check_distance(self,roi_size=15):
        while True:
            color_image,depth_image=self.get_data()

            #查看测距是否准确,随机取几个点,然后进行测距,看看效果
            color_map=self.get_color_map(depth_image,10000)

            #获取图像中心点
            h,w=depth_image.shape
            h=int(h/2)
            w=int(w/2)

            #生成一个区域进行测距
            xyz_image=self.get_xyz_image()
            roi_w=roi_size
            roi_h=roi_size
            middle_roi=xyz_image[h-roi_h:h+roi_h,w-roi_w:w+roi_w]#得到中心区域的xyz值
            middle_roi=middle_roi.reshape(-1,3)

            #对选取区域求平均之后去除掉方差以外的值
            mean=np.mean(middle_roi[:,2])
            std=np.std(middle_roi[:,2])
            origin_number=len(middle_roi)
            correct_middle_roi=abs(middle_roi[:,2]-mean)<0.8*std#在其内部的roi值
            middle_roi=middle_roi[correct_middle_roi]
            new_number=len(middle_roi)
            print("选取剩余0.8个方差之后的值有:{},剩余值占原来的{:.2f}%".format(new_number,new_number/origin_number*100))

            #得到最终的测试距离
            mean_distance=np.mean(middle_roi[:,2])#获取正确的xyz值

            #最后输出测距距离
            print("中心的距离为:",mean_distance)
            color_map[h-roi_h:h+roi_h,w-roi_w:w+roi_w]=(0,0,255)

            cv.imshow("depth_image",depth_image)
            cv.imshow("color_map",color_map)
            cv.waitKey(0)

    def read_depth_callback(self,event,x,y,flags,param):

        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.read_image_dpeth[y,x]
                print("Depth is :",hsv)
            except:
                return


def get_images_example():
    camera=RS(resolution='640x480',align_to_depth=False,use_filter=True)
    while True:
        color_image,depth_image=camera.get_data()
        color_map=camera.get_color_map()
        cv.imshow("color_image",color_image)
        cv.imshow("color_map",color_map)
        cv.waitKey(1)


def capture_images(save_path="MyFile"):
    camera=RS(resolution='640x480',align_to_depth=False,use_filter=True)
    save_number=0

    while True:
        color_image,depth_image=camera.get_data()
        camera.read_image_dpeth=depth_image.copy()
        color_map=camera.get_color_map()
        cv.imshow("color_image",color_image)
        cv.imshow("color_map",color_map)
        cv.namedWindow("Read_Depth")
        cv.imshow("Read_Depth",color_image)
        cv.setMouseCallback("Read_Depth",camera.read_depth_callback)

        input_info=cv.waitKey(1)
        if input_info==115:#'s'保存
            cv.imwrite(save_path+"/{}_color.png".format(save_number),color_image)
            cv.imwrite(save_path+"/{}_depth.png".format(save_number),depth_image)#OpenCV3可以直接保存深度图,2不行
            print("Save_image {}".format(save_number))
            save_number=save_number+1

def check_depth():
    camera=RS()
    depth=cv.imread("MyFile/0_depth.png",cv.IMREAD_UNCHANGED)
    color_map=camera.get_color_map(depth)
    cv.imshow("color_map",color_map)
    cv.waitKey(0)




if __name__ == '__main__':
    # get_images_example()
    capture_images(save_path="each_ojbect")
    # check_depth()