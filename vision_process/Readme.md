# Readme

## 1. 快速使用

1. python包安装

     ```
     pip install torch==1.1.0 torchvision==0.2.0
     ```

     (其余包缺少时,注意要安装python2的版本)

     

2. 在vision_process中的DenseFusion_Lib中创建models文件夹,下载DenseFusion的网络模型:

     ```
     链接: https://pan.baidu.com/s/1d-px7w1C6uXhLJAKTByJ5g  密码: ngam
     ```

     

3. 在vision_process中的SegNet_Lib中创建models文件夹,下载SegNet的网络模型:

     ```
     链接: https://pan.baidu.com/s/1LRlL_PEil8cAWvyLDXbSWw  密码: ng61
     ```

     

3. 在执行代码时,直接调用ocrtoc_solution的robot_control.py中的objects类即可获取Pose

     objects的定义中,如果设置get_pose_from_gazebo=False,则从DenseFusion中调用.

     调用前,需要启动DenseFusionInfer.py

     ```
     rosrun vision_process DenseFusionInfer.py
     ```

     - 显卡在1060上测试可行,但是1050Ti可能显存不够导致CUDA报错误





