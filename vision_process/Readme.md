# Readme

## 1. 快速使用

1. python包安装

     ```
     pip install torch==1.1.0 torchvision==0.2.0
     pip install pyrsistent==0.15.6
     pip intall open3d==0.8.0.0
     pip install scipy pypng
     ```

     (其余包缺少时,注意要安装python2的版本)

     

2. 在vision_process/scripts/DenseFusion_Lib/models和vision_process/scripts/SegNet_Lib/models两个文件夹中添加对应的网络模型,保存在Newest的文件夹中(最好随时联系蒋俊南,获取最新的识别网络)

     ```
     链接: https://pan.baidu.com/s/1MkONCS5DiYbzXIt8QVfZVg  密码: d55w
     ```

     - Gazebo和SegNet的网络模型不同,下载的时候需要注意

     

3. 在执行代码时,直接调用ocrtoc_solution的robot_control.py中的objects类即可获取Pose

     objects的定义中,如果设置get_pose_from_gazebo=False,则从DenseFusion中调用.

     调用前,需要启动DenseFusionInfer.py

     ```
     rosrun vision_process DenseFusionInfer.py
     ```

     - 显卡在1060上测试可行,但是1050Ti可能显存不够导致CUDA报错误







## 2. 识别类说明

识别主要是DenseFusion_Detector类,功能性函数有:

```python
if __name__ == '__main__':
    densefusion_Detector=DenseFusion_Detector(init_node=True)
    # densefusion_Detector.pub_pose_array()#发布所有物体Pose
    # densefusion_Detector.check_densefusion("1-1")#确定场景识别精度
    # densefusion_Detector.see_detect_result(debug=True)#用于查看这个场景的识别结果
    # change_pth()#更改网络zip包
```

- pub_pose_array(): 发布物体Pose的Topic
- check_densefusion():指定场景,(需要开启sapien仿真器),获取DenseFusion在这个场景中的识别精度
- see_detect_result():查看所有物体识别的pose









