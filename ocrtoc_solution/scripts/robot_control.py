#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
import numpy as np
import time
import math
from trac_ik_python.trac_ik import IK
from urdf_parser_py.urdf import URDF
from kdl_parser import kdl_tree_from_urdf_model
import PyKDL

# msgs
from sensor_msgs.msg import JointState
from gazebo_msgs.msg  import ModelStates
from control_msgs.msg import GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
import quaternion
from kdl_conversions import *
#transform
import tf.transformations as trans_tools

class Robot(object):
    def __init__(self, init_node=False, node_name='test', base_link ="world", tip_link = "robotiq_2f_85_ee_link"):
        if init_node:
            rospy.init_node(node_name)
        # robot state
        self._x, self._dx, self._q, self._dq = None, None, None, None
        self._J, self._p = None, None
        self._reachable=True
        
        # gripper
        self._gripper_is_grasped = 0.
        self._gripper_width = 0.07
        self.gripper_cmd_pub=rospy.Publisher(rospy.resolve_name('gripper_controller/gripper_cmd/goal'),GripperCommandActionGoal,queue_size=10)

        # load robot kdl tree
        self._base_link = base_link
        self._tip_link = tip_link
        self._urdf = URDF.from_parameter_server(key='robot_description')
        self._kdl_tree = kdl_tree_from_urdf_model(self._urdf)
        self._arm_chain = self._kdl_tree.getChain(self._base_link, self._tip_link)
        self._joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self._num_jnts = len(self._joint_names)
        self._stop = False

        # KDL  forward kinematics
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)

        self._jac_calc = PyKDL.ChainJntToJacSolver(self._arm_chain)
        # trac_ik inverse kinematics
        urdf_str = rospy.get_param('/robot_description')
        self.trac_ik_solver = IK(self._base_link, self._tip_link, urdf_string=urdf_str)  


        # subscribe robot joint states
        self.robot_state_sub = rospy.Subscriber("joint_states", JointState, self.robot_state_cb, queue_size=1000 )
        # send pose command to C++ driver
        self.pose_cmd_pub = rospy.Publisher(
                    rospy.resolve_name('ur_command'),
                    Float64MultiArray, queue_size=10)
        # send joints to arm
        self.joint_cmd_pub = rospy.Publisher(
            rospy.resolve_name('arm_controller/command'),
            JointTrajectory, queue_size=10)

        #transform world to base
        self.trans_world2base=np.array([[  7.96326711e-04,  -9.99999683e-01,   0.00000000e+00,
                  0.00000000e+00],
               [  9.99999683e-01,   7.96326711e-04,   0.00000000e+00,
                  2.40000000e-01],
               [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
                  5.00000000e-03],
               [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                  1.00000000e+00]])

        #绕x轴的旋转180度矩阵
        self.trans_x180Matrix=np.array([[1,0,0,0],
                                      [0,-1,0,0],
                                      [0,0,-1,0],
                                      [0,0,0,1]])

    #####################################机械臂控制接口##############################
    def forward_kinematics(self, q):
        # input q , numpy array (6,)
        # return [px,py,pz,qx,qy,qz,qw]
        kdl_jnts = joint_to_kdl_jnt_array(q)

        end_frame = PyKDL.Frame()
        self._fk_p_kdl.JntToCart(kdl_jnts, end_frame)

        pos = end_frame.p
        quat = PyKDL.Rotation(end_frame.M).GetQuaternion()  # x y z w
        jac = PyKDL.Jacobian(self._num_jnts)

        self._jac_calc.JntToJac(kdl_jnts, jac)
        jac_array = kdl_matrix_to_mat(jac)
        return np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]), jac_array

    def inverse_kinematics(self, pos, quat, seed=None):
        
        if seed is None:
            seed  = self._q
        result = self.trac_ik_solver.get_ik(seed,
                                            pos[0], pos[1], pos[2],  # X, Y, Z
                                            quat[0], quat[1], quat[2], quat[3])     # QX, QY, QZ, QW     
         
        return np.array(result) if result is not None else None

    def robot_state_cb(self, data):
        
        q = np.zeros(self._num_jnts)
        dq = np.zeros(self._num_jnts)
        # assign joint angle and velocity by joint name
        for i in range(self._num_jnts):
            for j in range( len(data.name) ):
                if data.name[j] == self._joint_names[i]:
                    q[i] = data.position[j]
                    dq[i] = data.velocity[j]
                    break
        self._q = q
        self._dq = dq
        jac = PyKDL.Jacobian(self._num_jnts)
        self._x, self._J = self.forward_kinematics(self._q)
        self._dx = np.dot(self._J, self._dq.reshape(-1,1)).flatten()
        self._p = quaternion.matrix_from_quaternion(self._x[3:], pos =self._x[:3])

    # motion planning
    def move_to_joint(self, joint, t):
        # send joint to robot
        joint_cmd = JointTrajectory()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.joint_names = self._joint_names

        if self._q is not None:
            # q0 current joints
            q0 = JointTrajectoryPoint()
            q0.positions = self._q.tolist()
            q0.velocities = [0.]*6
            q0.time_from_start.secs = 0.01
            # joint_cmd.points.append(q0)

        #q1 target joints
        q1 = JointTrajectoryPoint()
        if type(joint) is np.ndarray:
            joint_list = joint.tolist()
        else:
            joint_list= joint
        q1.positions = joint_list
        q1.velocities = [0.]*6
        # q1.time_from_start.secs = t        
        q1.time_from_start = rospy.Duration(t)
        joint_cmd.points.append(q1)
        self.joint_cmd_pub.publish(joint_cmd)
        rospy.sleep(t)

    def move_to_frame(self, x, t,seed=None):
        # x [x,y,z, qx, qy, qz, qw,]
        # t, time for execution
        # qd = Float64MultiArray()
        # qd.data = np.concatenate([x, np.array([t])])
        # self.pose_cmd_pub.publish(qd)

        qd = self.inverse_kinematics(x[:3],x[3:])
        if qd is None:
            rospy.logerr('qd is None, Inverse kinematics fail!!!')

        else:
            self.move_to_joint(qd, t )

    def home(self, t=10):
        p = np.array([ 1.55986781, -2.1380509 ,  2.49498554, -1.93086818, -1.5671494 , 0])
        self.move_to_joint(p,t)

    def getpose_home(self, t=1):
        """
        DenseFusion获取抓取点的时候需要运动到这个位置,避免爪子出现在摄像头中
        :param t:
        :return:
        """
        p = np.array([ 1.55986781, -2.1380509 ,  1.5, -1.93086818, -1.5671494 , 0])
        self.move_to_joint(p,t)

    def sin_test(self, delta_z = 0.2, T = 20.):
        # sin movement test, z = A*sin(w*t)
        
        x0 = np.copy(self._x)
        print(x0)
        t0 = rospy.get_time()
        freq = 10
        r = rospy.Rate(freq)
        while not rospy.is_shutdown():
            t = rospy.get_time() - t0
            xd = np.copy(x0)
            xd[2] = x0[2] +  delta_z * np.sin(2*np.pi/T* t )
            self.move_to_frame(xd, 1./freq )
            r.sleep()


        return True

    def slerp(self, v0, v1, t_array):
        """Spherical linear interpolation."""
        # from https://en.wikipedia.org/wiki/Slerp
        # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
        t_array = np.array(t_array)
        v0 = np.array(v0)
        v1 = np.array(v1)
        dot = np.sum(v0 * v1)

        if dot < 0.0:
            v1 = -v1
            dot = -dot
        
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
            return (result.T / np.linalg.norm(result, axis=1)).T
        
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * t_array
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])
        
    def motion_generation(self, poses, vel=0.2, intepolation='linear',debug=False):
        # poses : (n,7) array, n: num of viapoints. [position, quaternion]
        poses = np.concatenate([self.x.reshape(1,-1), poses],axis=0) # add current points
        keypoints_num = poses.shape[0]

        path_length = 0
        for i in range(keypoints_num - 1):
            path_length += np.linalg.norm( poses[i,:3] - poses[i+1,:3])
        path_time = path_length / vel

        if debug:
            print poses[:,:3]
            print 'keypoints num = ', keypoints_num
            print 'Total path time : ', path_time, "s,  path length", path_length,'m'

        sample_freq = 20  # 20Hz
        joint_seed = self.q
        if not self._stop:
            for i in range(keypoints_num-1):
                path_i =  np.linalg.norm( poses[i,:3] - poses[i+1,:3])
                # print(path_i)
                sample_num = int(path_i / vel * sample_freq +1)

                if debug:
                    print(path_i)
                    rospy.loginfo("start to go the " + str(i+1)+  "-th point: " + " x="+str(poses[i+1,0]) + " y="+str(poses[i+1,1])
                            + " z="+str(poses[i+1,3])+" time: " + str(path_i / vel) +"s")
                if intepolation=='linear':
                    pos = np.concatenate((np.linspace(poses[i,0],poses[i+1,0], num=sample_num).reshape(-1,1),
                                        np.linspace(poses[i,1],poses[i+1,1], num=sample_num).reshape(-1,1),
                                        np.linspace(poses[i,2],poses[i+1,2], num=sample_num).reshape(-1,1) ), axis=1)                # print 
                ori = self.slerp(poses[i,3:], poses[i+1,3:] , np.array(range(sample_num+1), dtype=np.float)/sample_num    )
                for j in range(sample_num):
                    target_x = np.concatenate((pos[j,:], ori[j,:])   )
                    self.move_to_frame(target_x, 1./sample_freq  )
                    rospy.sleep(1./sample_freq)
        return True

    def gripper_control(self,angle,force,debug=False):
        gripper_cmd=GripperCommandActionGoal()
        gripper_cmd.goal.command.position=angle
        gripper_cmd.goal.command.max_effort=force
        self.gripper_cmd_pub.publish(gripper_cmd)
        if debug:
            rospy.loginfo("Pub gripper_cmd")
        rospy.sleep(1.0)

    #####################################抓取任务执行##############################
    def transform_world2base(self,world_pose):
        """
        Transform the pose in world frame to baselink frame
        :param world_pose: world_pose[x,y,z,qx,qy,qz,qw]
        :return:baselink_pose
        """
        worldpose_Matrix=trans_tools.quaternion_matrix(world_pose[3:])
        worldpose_Matrix[0:3,3]=np.array(world_pose[:3]).T

        basepose_Matrix=self.trans_world2base.dot(worldpose_Matrix)
        rot=trans_tools.quaternion_from_matrix(basepose_Matrix)
        trans=basepose_Matrix[0:3,3].T

        base_pose=np.hstack([trans,rot])
        return base_pose

    def get_pickpose_from_pose(self,pose):
        """
        送入物体pose,获取所对应的抓取pose
        直接认为抓取pose就是物体朝向反过来即可
        之后还需要注意,如果物体的朝向是向下的,则从背面抓取(这个先不解决)
        :param pose:直接乘上一个围绕x轴转180度的变换矩阵即可(当然也可以顺着y轴,但是先不管y轴)
        :return:
        """
        pose_Matrix=trans_tools.quaternion_matrix(pose[3:])
        pose_Matrix[0:3,3]=np.array(pose[:3].T)

        #乘上变换矩阵
        grasp_Matrix=pose_Matrix.dot(quaternion.euler_matrix(0,math.pi,0))
        rot=trans_tools.quaternion_from_matrix(grasp_Matrix)
        trans=grasp_Matrix[0:3,3].T
        converted_pose=np.hstack([trans,rot])
        return converted_pose

    def move_updown(self,pose,grasp=False,fast_vel=0.4,slow_vel=0.1):
        """
        从上方运动到物体的给定位置
        :param pose:
        :param grasp: 落下去后抓取或者释放
        :return:
        """
        upper_pose=pose.copy()
        upper_pose[2]=upper_pose[2]+0.2#抬高20cm
        #从上往下运动
        self.motion_generation(upper_pose[np.newaxis,:],vel=fast_vel)
        self.motion_generation(pose[np.newaxis,:],vel=slow_vel)
        if grasp:
            # self.grasp_slowly(0.5,force=1)
            self.gripper_control(angle=0.5,force=1)
        else:
            self.gripper_control(angle=0,force=1)
        time.sleep(1)
        self.motion_generation(upper_pose[np.newaxis,:],vel=slow_vel)


    def grasp_slowly(self,target_angle,force=0,points=20):
        """
        尝试一点点地缩小抓取范围,看看能不能抓上
        :param target_angle: 目标角度
        :return:
        """
        middle_angle=np.linspace(0,target_angle,num=points)
        for angle in middle_angle:
            self.gripper_control(angle,force)


    @property
    def x(self):
        """
        Position and orientation (quaternion) of the end-effector in base frame
        :return: [position, orientation] (7,)
        """
        return self._x

    @property
    def q(self):
        """
        Joint angles
        :return: [q ] (7,)
        """
        return self._q

    @property
    def dq(self):
        """
        Joint velocities
        :return: [q ] (7,)
        """
        return self._dq

    @property
    def dx(self):
        """
        Velocity of the defined end-effector in base frame

        :return: [velocity, angular velocity] (6,)
        """
        if self._dx is None and self._J is not None:
            self._dx = np.dot(self.J, self.dq)

        return self._dx

    @property
    def J(self):
        """
        jacobian of end-effector in base frame
        :return:  (6, 7)
        """
        return self._J

    @property
    def p(self):
        """
        Transformation matrix of end-effector
        :return: (4, 4)
        """
        if self._p is None:
            self._p = np.eye(4)

        return self._p

class Objects(object):
    def __init__(self, init_node = False,get_pose_from_gazebo=False):
        """
        获取物体姿态参数
        :param init_node:初始化node
        :param get_pose_from_gazebo: True则从Gazebo中获取,False则从DenseFusion中获取
                                     DenseFusion的话,需要rosrun vision_process DenseFusionInfer.py
        """
        if init_node:
            rospy.init_node('object_positions')

        self._names, self._nums = None, None
        self._x = None
        # subscribe object position and orientation
        ## todo, we need to do object localization by the cameras
        ## now, I use the Gazebo topic to get them in world frame.
        if get_pose_from_gazebo:
            # self.objects_state_sub = rospy.Subscriber("gazebo/model_states", ModelStates, self.objects_state_cb, queue_size=10)#读取gazebo信息
            self.objects_state_sub = rospy.Subscriber("/sapien/get_model_state", ModelStates, self.objects_state_cb, queue_size=10)#读取sapien信息
        else:
            self.DenseFuion_result_sub=rospy.Subscriber("/poseinworld",ModelStates,self.objects_state_cb,queue_size=10)

        self.update_pose=False#用于决定是否更新Pose

    def objects_state_cb(self,data):
        if self.update_pose:#在获取位姿的时候为True,获取完成之后为False
            self._names = data.name
            self._nums = len(self._names)
            self._x = np.zeros([self._nums, 7])  #  (n,7) numpy array
            for i in range(self._nums):
                pose = data.pose[i]
                self._x[i,:3] = np.array([pose.position.x, pose.position.y, pose.position.z ])
                self._x[i,3:] = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                     pose.orientation.w ] )
    @property
    def x(self):
        """
        Position and orientation (quaternion) of all objects in world frame
        :return: [position, orientation] (n,7)
        """
        return self._x
    @property
    def names(self):
        """
        :return: [name1, name2...] )
        """
        return self._names

    def get_pose(self,debug=False):
        self.update_pose=True
        self._x=None#清除以前的x的Pose
        rate=rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._x is not None:
                if debug:
                    print("object_name is:")
                    print(self._names)
                    print ("Pose is")
                    print(self._x)
                    break
                else:
                    print("Already Updata Objects Pose")
                    break
            else:
                print("[Warning],can't get the pose")
            rate.sleep()

        self.update_pose=False



#####################################测试函数部分##############################
def move_to_object():
    """
    这里面基于DenseFusion识别到的结果,运动到每个物体的旁边
    :return:
    """
    robot=Robot(init_node=True)
    robot.getpose_home(3)
    objects=Objects(get_pose_from_gazebo=False)#从Gazebo中获取Pose
    while not rospy.is_shutdown():
        robot.getpose_home()
        print("Ready to get the picture")
        objects.get_pose()
        for i,pose in enumerate(objects.x):
            robot.home(t=1)
            name=objects.names[i]
            print("name is",name)
            if name=='robot' or name=="ground":
                continue
            grasp_pose=robot.get_pickpose_from_pose(pose)#Z轴翻转获取物体的抓取Pose
            upper_pose=grasp_pose.copy()
            upper_pose[2]=upper_pose[2]+0.2#抬高30cm
            print("Traget is {},it's Pose is {}".format(objects.names[i],pose))
            #从上往下进行抓取
            robot.motion_generation(upper_pose[np.newaxis,:],vel=0.4)
            robot.motion_generation(grasp_pose[np.newaxis,:])
            robot.motion_generation(upper_pose[np.newaxis,:])
            print("Move to {}".format(objects.names[i]))
        break

def test_gripper():
    """
    这里面尝试进行物体抓取
    :return:
    """
    robot=Robot(init_node=True)
    objects=Objects(get_pose_from_gazebo=True)#从Gazebo中获取Pose
    while not rospy.is_shutdown():
        robot.getpose_home()
        print("Ready to get the Pose")
        objects.get_pose()
        for i,pose in enumerate(objects.x):
            robot.home(t=1)
            name=objects.names[i]
            print("name is",name)
            if name=='robot' or name=="ground":
                continue
            print("Target Pose is",pose)
            grasp_pose=robot.get_pickpose_from_pose(pose)#Z轴翻转获取物体的抓取Pose

            print("Grasp Pose is",grasp_pose)
            robot.gripper_control(angle=0,force=0)
            robot.move_updown(grasp_pose,grasp=True)
            robot.home(t=1)
            robot.move_updown(grasp_pose,grasp=False)
            print("Move to {}".format(objects.names[i]))
        break


def test_sapien():
    robot=Robot(init_node=True)
    while not rospy.is_shutdown():
        robot.getpose_home()
        print("Ready to get the Pose")
        robot.home(t=1)
        #更改一下pose
        pose=np.array([0.16,-0.14,0.01,0.48,0,0,0.87])
        grasp_pose=robot.get_pickpose_from_pose(pose)#Z轴翻转获取物体的抓取Pose
        print("Grasp Pose is",grasp_pose)
        robot.gripper_control(angle=0,force=0)
        robot.move_updown(grasp_pose,grasp=True)
        robot.home(t=1)
        robot.move_updown(grasp_pose,grasp=False)

        break


def move_home():
    """
    机械臂撞东西之后返回原来的状态
    :return:
    """
    robot=Robot(init_node=True)
    while not rospy.is_shutdown():
        robot.getpose_home(3)
        robot.home(t=1)


if __name__ == '__main__':
    test_sapien()
    # test_gripper()
    # move_home()





