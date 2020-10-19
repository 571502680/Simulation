#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
这里封装DenseFusion的使用函数,从而让整体可以前向推理
Log:2020.9.14:
    开始进行DenseFusion封装

Log:2020.9.15:
    整合了代码,避免代码太过复杂.
    SegNet采用np.argmax进行统计,与此同时,设置了一定的区域分割,从而确保没有太多其他参数

Log:2020.9.27:
    基于更改过的Read_Data和Make_Data,进行DenseFusion的执行

Log:2020.10.19:
    重构识别代码,开始尝试使用多视角进行Pose识别任务,尝试是否能够提升所对应的效果
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
#机器人控制库
import robot_control


