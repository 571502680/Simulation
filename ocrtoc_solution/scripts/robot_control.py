import rospy
import numpy as np
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


class Robot(object):
    def __init__(self, init_node=False, node_name='test', base_link ="world", tip_link = "robotiq_2f_85_ee_link"):
        if init_node:
            rospy.init_node(node_name)
        # robot state
        self._x, self._dx, self._q, self._dq = None, None, None, None
        self._J, self._p = None, None
        
        # gripper
        self._gripper_is_grasped = 0.
        self._gripper_width = 0.07

        # load robot kdl tree
        self._base_link = base_link
        self._tip_link = tip_link
        self._urdf = URDF.from_parameter_server(key='robot_description')
        self._kdl_tree = kdl_tree_from_urdf_model(self._urdf)
        self._arm_chain = self._kdl_tree.getChain(self._base_link, self._tip_link)
        self._joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self._num_jnts = len(self._joint_names)

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
        seed = self._q
        if seed is None:
            seed = [1.5621, -2.12, 1.72, -1.1455, -1.57, 0]
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

    def move_to_frame(self, x, t):
        # x [x,y,z, qx, qy, qz, qw,]
        # t, time for execution
        # qd = Float64MultiArray()
        # qd.data = np.concatenate([x, np.array([t])])
        # self.pose_cmd_pub.publish(qd)

        qd = self.inverse_kinematics(x[:3],x[3:])
        if qd is None:
            rospy.logerr('qd is None, Inverse kinematics fail!!!')
        else:
            self.move_to_joint(qd, t)

    def home(self, t=10):

        p = np.array([  1.56166289e+00 , -2.20212942e+00,   2.10209237e+00,  -1.44570209e+00,
  -1.57076397e+00,  -1.19897808e-03])
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

    def motion_generation(self, poses, vel=0.2, intepolation='linear'):
        # poses : (n,7) array, n: num of viapoints. [position, quaternion]
        return True









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

class objects(object):
    def __init__(self, init_node = False ):
        if init_node:
            rospy.init_node('object_positions')

        self._names, self._nums = None, None
        self._x = None
        # subscribe object position and orientation
        ## todo, we need to do object localization by the cameras
        ## now, I use the Gazebo topic to get them in world frame.
        self.objects_state_sub = rospy.Subscriber("gazebo/model_states", ModelStates, self.objects_state_cb, queue_size=1000 )

    def objects_state_cb(self,data):
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



