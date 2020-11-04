"""
这里面使用神经网络与传统方法结合进行矩形框的输出
"""
import glob
import cv2 as cv
import numpy as np

from Center_DetectClass import CenterDetect


class GraspwithID:
    def __init__(self,images_path=None):
        if images_path is None:
            self.images_path="offical_dataset"
        else:
            self.images_path=images_path

        #读取图片
        self.read_image_depth=None

    ##############################读取图片函数##############################
    def see_images(self):
        bgr_paths=glob.glob(self.images_path+"/*_color.png")
        for bgr_image in bgr_paths:
            print("当前看的图片是:",bgr_image)
            #获取图片信息
            image_number=bgr_image.split("/")[-1]
            image_number=image_number.split("_")[0]
            depth_path=self.images_path+"/{}_depth.png".format(image_number)

            #获取深度和图片
            image=cv.imread(bgr_image)
            depth=cv.imread(depth_path,cv.IMREAD_UNCHANGED)#深度为mm制
            self.read_image_depth=depth.copy()
            ROI=cv.inRange(depth,0,1500)#对于太远的去掉,省得看的并不明显
            depth=depth*ROI/255
            cv_image=depth.copy()
            cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
            cv_image=cv_image.astype(np.uint8)
            color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)

            #读取深度图参数,确定比较好的深度图
            cv.namedWindow("Read_Depth",cv.WINDOW_NORMAL)
            cv.imshow("Read_Depth",image)
            cv.setMouseCallback("Read_Depth",self.read_depth_callback)

            #最后进行图片展示
            cv.imshow("color_map",color_map)
            cv.imshow("image",image)
            cv.waitKey(0)

    def get_images(self,image_number):
        """
        :param image_number:
        :return:
        """
        image=cv.imread(self.images_path+"/{}_color.png".format(image_number))
        depth=cv.imread(self.images_path+"/{}_depth.png".format(image_number),cv.IMREAD_UNCHANGED)
        self.read_image_depth=depth.copy()
        return image,depth

    def read_depth_callback(self,event,x,y,flags,param):

        if event==cv.EVENT_MOUSEMOVE:
            try:
                hsv=self.read_image_depth[y,x]
                print("HSV is :",hsv)
            except:
                return

    ##############################基于深度图进行图像切割##############################
    def generate_kernel(self,x,y):
        return np.ones((x,y),np.uint8)

    def get_roi(self,depth_range,depth_image,image,see_depth=False,debug=False):
        """
        基于depth_range获取深度图图像
        :param depth_range: cv.inRange的参数
        :param depth_image: 深度图
        :param image: BGR图,用于进行debug
        :param see_depth: 是否查看图片的深度值
        :param debug: 是否查看color_map用于调参
        :return:
        """

        #查看图片
        if debug:
            color_depth=depth_image.copy()
            cv.normalize(color_depth,color_depth,255,0,cv.NORM_MINMAX)
            color_depth=color_depth.astype(np.uint8)
            color_map=cv.applyColorMap(color_depth,cv.COLORMAP_JET)
            cv.imshow("color_map",color_map)

        #进行图片分割
        if see_depth:
            cv.namedWindow("Read_Depth",cv.WINDOW_NORMAL)
            cv.imshow("Read_Depth",image)
            cv.setMouseCallback("Read_Depth",self.read_depth_callback)

        ROI=cv.inRange(depth_image,0,depth_range)

        #进行ROI大小选择
        ROI=cv.morphologyEx(ROI,cv.MORPH_ERODE,self.generate_kernel(20,10))#用于缩小,从而避免东西变少了


        if debug:
            cv.imshow("ROI",ROI)
            cv.waitKey(0)
        return ROI

    def get_roi_singleobj(self,depth_range,depth_image,image,see_depth=False,debug=False):
        """
        基于depth_range获取深度图图像
        :param depth_range: cv.inRange的参数
        :param depth_image: 深度图
        :param image: BGR图,用于进行debug
        :param see_depth: 是否查看图片的深度值
        :param debug: 是否查看color_map用于调参
        :return:
        """

        #查看图片
        if debug:
            color_depth=depth_image.copy()
            cv.normalize(color_depth,color_depth,255,0,cv.NORM_MINMAX)
            color_depth=color_depth.astype(np.uint8)
            color_map=cv.applyColorMap(color_depth,cv.COLORMAP_JET)
            cv.imshow("color_map",color_map)

        #进行图片分割
        if see_depth:
            cv.namedWindow("Read_Depth",cv.WINDOW_NORMAL)
            cv.imshow("Read_Depth",image)
            cv.setMouseCallback("Read_Depth",self.read_depth_callback)

        #这里可以使用很大的闭操作



        ROI=cv.inRange(depth_image,0,depth_range)

        #进行ROI大小选择
        ROI=cv.morphologyEx(ROI,cv.MORPH_CLOSE,self.generate_kernel(10,10))

        if debug:
            cv.imshow("ROI",ROI)
            cv.waitKey(0)
        return ROI

    def process_rotate_rect(self,rotate_rect):
        """
        用于处理旋转矩形,得到一个长短边正确的旋转矩形
        :param rotate_rect:
        :return:
        """
        center,wh,xita=rotate_rect
        width=wh[0]
        height=wh[1]
        if width>height:
            xita=xita+90
            correct_width=height
            correct_height=width
            return center,(correct_width,correct_height),xita
        else:
            return rotate_rect

    def show_pre_grasp_rect(self,ROI,image,debug=False,see_image=False,draw_bbox_flag=False):
        """
        这里面展示一张图里面所有的抓取矩形框
        之后机械臂能够运动的时候,需要再做精细的调整
        :param ROI:
        :param image:
        :param debug:
        :param see_image:
        :return:
        """
        #1:对ROI找轮廓
        contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour)<1500:
                # if debug:
                #     print("contour:{}".format(cv.contourArea(contour)))
                continue

            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确
            if debug:
                #绘制物体框
                draw_box=cv.boxPoints(rotate_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(image,[draw_box],0,(0,0,255),2)
                if draw_bbox_flag:
                    x,y,w,h=cv.boundingRect(contour)
                    cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

            #2:获取抓取矩形
            grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])

            #3:进行抓取矩形框绘制
            if debug:
                draw_box=cv.boxPoints(grasp_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(image,[draw_box],0,(0,255,0),3)

        if see_image:
            cv.imshow("grasprect_BGR_image",image)
            cv.waitKey(0)

    def get_grasp_fromroi(self,ROI_single):
        contours,_=cv.findContours(ROI_single,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        if len(contours)<1:
            print("[Warning] the ROI_single has no ojbect")
            return None

        if len(contours)>1:
            #针对这里面进行形态学操作,让他们变成一个
            ROI_single=cv.morphologyEx(ROI_single,cv.MORPH_CLOSE,self.generate_kernel(30,30))
            contours,_=cv.findContours(ROI_single,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            contours=sorted(contours,key=lambda data:len(data))

            contour=contours[-1]
            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确

            #2:获取抓取矩形
            grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])
            return grasp_rect


        if len(contours)==1:#只有一个的情况的解决
            for contour in contours:
                rotate_rect=cv.minAreaRect(contour)
                rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确

                #2:获取抓取矩形
                grasp_rect=(rotate_rect[0],(rotate_rect[1][0]+50,100),rotate_rect[2])
                return grasp_rect

    ##############################进行每个物品信息识别任务##############################
    def get_obj_info(self,ROI,image,depth,debug=False,see_image=False):
        #1:对ROI找轮廓
        contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        temp_array=np.zeros(ROI.shape,dtype=np.uint8)
        for i,contour in enumerate(contours):
            if cv.contourArea(contour)<1500:
                # if debug:
                #     print("contour:{}".format(cv.contourArea(contour)))
                continue
            #1.1:生成标注框
            rotate_rect=cv.minAreaRect(contour)
            rotate_rect=self.process_rotate_rect(rotate_rect)#确保长宽正确

            if debug:
                #绘制物体框
                draw_box=cv.boxPoints(rotate_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(image,[draw_box],0,(0,0,255),2)

            #1.2:单独切割出物体的轮廓
            temp_array=np.zeros(ROI.shape,dtype=np.uint8)
            cv.drawContours(temp_array,contours,i,255,thickness=-1)
            area=cv.contourArea(contour)

            depth_roi=depth*temp_array/255.0
            mean=np.sum(depth_roi)/area
            if debug:
                cv.putText(image,"{:.3f}".format(mean),tuple(draw_box[0]),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        if see_image:
            if len(contours)>1:
                cv.imshow("temp_array",temp_array)
            cv.imshow("image",image)
            cv.waitKey(0)

    ##############################带种类的抓取矩形框获取##############################
    def getgrasp_withid(self,results,depth,image=None,debug=False,see_image=False):
        """
        用于获取带种类的抓取id
        :param results: 网络识别结果
        :param depth: 深度图
        :param image: 颜色图
        :param debug:
        :param see_image:
        :return:
        """
        grasp_rect=[]

        #1:获取ROI
        ROI=self.get_roi(600,depth,image)
        contours,_=cv.findContours(ROI,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        #2:基于result进行ROI选择,并最终生成抓取矩形框,在image上显示
        for result in results:
            x1,y1,x2,y2,score,category=result
            if see_image:
                cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                cv.putText(image,"{}".format(category),(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),thickness=2)

            #基于矩形框进行ROI的二次分割
            temp_array=np.zeros(ROI.shape,dtype=np.uint8)
            temp_array[y1:y2,x1:x2]=255
            _,inverse_array=cv.threshold(temp_array,200,255,cv.THRESH_BINARY_INV)
            inverse_array=inverse_array.astype(np.int32)
            inverse_array=inverse_array*10

            temp_depth=depth.copy()
            temp_depth=temp_depth.astype(np.int32)
            temp_depth=temp_depth+inverse_array

            if debug:
                cv_image=temp_depth.copy()
                cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
                cv_image=cv_image.astype(np.uint8)
                color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
                cv.imshow("color_map",color_map)


            ROI_single=self.get_roi_singleobj(600,temp_depth,image)
            if debug:
                cv.imshow("ROI_single",ROI_single)

            grasp_rect=self.get_grasp_fromroi(ROI_single)
            if grasp_rect is not None:
                draw_box=cv.boxPoints(grasp_rect)
                draw_box=np.int0(draw_box)
                cv.drawContours(image,[draw_box],0,(0,0,255),3)



        if see_image:
            cv.imshow("image",image)
            cv.waitKey(0)


def see_images():
    graspwithID=GraspwithID()
    graspwithID.see_images()

def get_pre_grasp():
    """
    基于depth进行图像分割,得到分割图片,最终得到这张图的所有潜在抓取矩形框
    :return:
    """
    graspwithID=GraspwithID()
    for number in range(27):
        image,depth=graspwithID.get_images(number)
        ROI=graspwithID.get_roi(600,depth,image)
        cv.imshow("ROI",ROI)
        graspwithID.show_pre_grasp_rect(ROI,image,debug=True,see_image=True,draw_bbox_flag=True)

def get_objects_info():
    """
    基于得到的每个物体的框,进行物体信息提取
    :return:
    """
    graspwithID=GraspwithID("offical_dataset")
    for number in range(27):
        image,depth=graspwithID.get_images(number)
        ROI=graspwithID.get_roi(600,depth,image)
        cv.imshow("ROI",ROI)
        graspwithID.get_obj_info(ROI,image,depth,debug=True,see_image=True)

def get_id_grasp():
    """
    获取带id的抓取矩形框
    :return:
    """
    #1:对图片中的物体进行识别
    graspwithID=GraspwithID()
    center_detect=CenterDetect(model_path="/home/elevenjiang/Documents/Project/IROS_Pick/Code/2DDetect/AutoLabel/CenterNet_File/CenterNet_IROSPICK/models/model_last.pth",Task='ctdet')
    for number in range(297):
        image,depth=graspwithID.get_images(number)
        detections=center_detect.detect(image)#获取识别内容,
        results=center_detect.fileter_detections_allcategory(detections=detections)

        #3:对于每个物体的ROI进行一次清洗,最终解决问题
        graspwithID.getgrasp_withid(results,depth,image,see_image=True,debug=True)


if __name__ == '__main__':
    # get_pre_grasp()
    # see_images()
    # get_objects_info()
    get_id_grasp()

