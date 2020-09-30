#! /usr/bin/env python
import rospy
import actionlib
import numpy as np
import time
from gazebo_msgs.msg  import ModelStates
from control_msgs.msg import GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import robot_control as rc
from gazebo_msgs.srv import *
import quaternion
import tf.transformations as trans_tools
import math
'''
box1 pudding_box  default
box2 wood_block  default
box3 jenga  default

can1 potted_meat_can  need to adjust Pi/2
can2 master_chef_can   default




'''
modellist = [['bottle1','bleach_cleanser'],['bottle2','mustard_bottle'],['toy1','foam_brick'],['toy2','d_toy_airplane'],['can1','tomato_soup_can'],['can2','master_chef_can'],['box1','wood_block'],['box2','rubiks_cube'],['box3','gelatin_box'],['box4','pudding_box']]

modellist2 = [['box2','jenga']]
#list contains unusual object,need to find more details by running densefusion
Need_to_Adjust = {'gelatin_box':[0,0,0,0,math.pi,math.pi/2],'potted_meat_can':[0,0,0,0,math.pi,math.pi/2],'b_toy_airplane':[0,0,0,0,math.pi,math.pi/2],'mustard_bottle':[0,0,0,0,3*math.pi/4,math.pi/2],
                    'bleach_cleanser':[0,0,0,0,3*math.pi/4,math.pi/2],'red_car':[0,0,0,0,math.pi,math.pi/2],'green_car':[0,0,0,0,math.pi,math.pi/2],'correction_fuid':[0,0,0,0,math.pi,math.pi/2],
                    'pure_zhen':[0,0,0.02,0,math.pi,0],'conditioner':[0,0,0.02,0,math.pi,0]}
#all cups need to adjust

class Objects1(object):
    def __init__(self, init_node = False):
        self.state = None
        self.xyzqua = np.zeros([7])
        self.model = GetModelStateRequest()
        # subscribe object position and orientation
        self.get_state_service = rospy.ServiceProxy('/sapien/get_model_state', GetModelState)

    def get_model_poses(self,modelname,relativeentityname):
        self.model.model_name = modelname
        self.model.relative_entity_name = relativeentityname
        self.state = self.get_state_service(self.model)
        self.xyzqua[:3] = np.array([self.state.pose.position.x,self.state.pose.position.y, self.state.pose.position.z])
        self.xyzqua[3:] = np.array([self.state.pose.orientation.x, self.state.pose.orientation.y, self.state.pose.orientation.z,
                     self.state.pose.orientation.w ] )
        #self.xyz = (self.state.pose.position.x, self.state.pose.position.y, self.state.pose.position.z)
        #print(self.xyzqua)
        #print(self.xyz)

def get_pickpose_from_pose(pose, x=0,y=0,z=0,degreeR=0,degreeP=math.pi,degreeY=0):
    print(pose)
    euler = trans_tools.euler_from_quaternion(pose[3:])

    pose_Matrix=trans_tools.quaternion_matrix(pose[3:])
    move_xyz = pose[:3] + np.array([-x*math.sin(euler[2]*180/math.pi)-y*math.cos(euler[2]*180/math.pi),-y*math.sin(euler[2]*180/math.pi)-x*math.cos(euler[2]*180/math.pi),z])
    pose_Matrix[0:3,3]=np.array(move_xyz.T)
    print(pose_Matrix)
    grasp_Matrix=pose_Matrix.dot(quaternion.euler_matrix(degreeR,degreeP,degreeY))
    rot=trans_tools.quaternion_from_matrix(grasp_Matrix)
    trans=grasp_Matrix[0:3,3].T
    converted_pose=np.hstack([trans,rot])
    return converted_pose

rospy.init_node('test_motion_control')

robot = rc.Robot()

obj = Objects1()

rospy.sleep(1)
robot.getpose_home(1)
for model1 in modellist2: 
    obj.get_model_poses(model1[0],model1[1])

    pick_pose = obj.xyzqua
    if model1[1] in Need_to_Adjust:
        param = Need_to_Adjust.get(model1[1])
        grasp_pose = get_pickpose_from_pose(pick_pose,x=param[0],y=param[1],z=param[2],degreeR=param[3],degreeP=param[4],degreeY=param[5])
    else:
        grasp_pose=get_pickpose_from_pose(pick_pose)
    robot.gripper_control(angle=0,force=10)
    robot.move_updown(grasp_pose,grasp=False,fast_vel=0.4,slow_vel=0.2)
    robot.home(t=1)

rospy.sleep(4)

# print range(1,4)
# print(ur.q)
# print(ur.p)
# ur.sin_test()
# rospy.sleep(1.0)
# print(ur._num_jnts)

# %%
# print obj.x[1]



# %%
