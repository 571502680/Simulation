#!/usr/bin/env python

from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint
import rospy
import actionlib
import math
from control_msgs.msg import GripperCommandActionGoal
import control_msgs.msg
from trac_ik_python.trac_ik import IK
from control_msgs.msg import GripperCommandActionGoal
from tf.transformations import quaternion_from_euler, quaternion_slerp

space = 0.1 #set the speed
ik_solver = IK("base_link", "wrist_3_link")
#########################################################
#还没有添加subscriber读取current state这个功能           #
#，我在这儿遇到了点麻烦。不过大体运动规划应该就是这样了。  #
#到时候直接搬进commit solution的user code部分            #
#########################################################


########################################
#主函数为motionplan，输入为pick，place，#
#高度，开合角和力度。输出为运动指令。   #
#                                     #
#######################################


'''
start = [0.24,-0.132,0.99,-1.57,0,0]
seed_state = [0.0] * ik_solver.number_of_joints
seed_state = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14]
pathtopick = [[seed_state[0], seed_state[1], seed_state[2], seed_state[3], seed_state[4], seed_state[5]]]
pick_pos = [0.2, 0.25, 0.315,-1.57,0,0]
none = [-0.205,0.4,0.4,-1.57,0,0]
'''
seed_state = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14] #初始关节角度
curr_posi = [-0.13,0.25,0.98,-1.57,0,0] #初始末端位置（可能有误）
def gripper_control(m,n):

        gripper_cmd = GripperCommandActionGoal()
        gripper_cmd.goal.command.position = m
        gripper_cmd.goal.command.max_effort = n
        gripper_cmd_pub.publish(gripper_cmd)
        rospy.loginfo("Pub gripper_cmd")
        rospy.sleep(1.0)


def distcal(start, end):
	dist_sqr = (start[0] - end[0])**2 + (start[1] - end[1])**2 + (start[2]-end[2])**2 
	dist = dist_sqr**0.5 #计算两目标点之间的距离
	return dist
def interpolation(point1, point2, lin_steps):   #进行插值
	waypoints_x = np.linspace(point1[0],point2[0],lin_steps)
	waypoints_y = np.linspace(point1[1],point2[1],lin_steps)	
	waypoints_z = np.linspace(point1[2],point2[2],lin_steps)
	way_xyz = map(list, zip(waypoints_x,waypoints_y,waypoints_z))
	fracs = np.linspace(0,1,lin_steps)
	point1_qua = quaternion_from_euler(point1[3],point1[4],point1[5])
	point2_qua = quaternion_from_euler(point2[3],point2[4],point2[5])
	way_quaternion = []
	for frac in fracs:
		new_qua = list(quaternion_slerp(point1_qua, point2_qua, frac))
		way_quaternion.append(new_qua)
	return way_xyz, way_quaternion


def build_trajectory(start, finish, curr_pos):  #curr_pos是'start'位置所对应的关节角度，第一次可订阅joint_state获得。
	'''
        if start is None:  # if given one pose, use current position as start
            start = [0.24,-0.132,0.99,-1.57,0,0]
	'''
	dist = distcal(start, finish)
	print(dist)
        steps = int(math.ceil(dist / space))
	way_xyz,way_quaternion = interpolation(start, finish, steps)
	print(way_xyz)
	print(way_quaternion)
	waytogo = [curr_pos]
	for i in range(0,len(way_xyz)):
		current_pose = waytogo[i]
		
		joint_space = ik_solver.get_ik(waytogo[i],
                way_xyz[i][0], way_xyz[i][1],way_xyz[i][2],  # X, Y, Z
       	         way_quaternion[i][0],way_quaternion[i][1], way_quaternion[i][2],way_quaternion[i][3])  # QX, QY, QZ, QW
		print(joint_space)
		waytogo.append(joint_space)
        return waytogo

def motionplan(pick_pos, place_pos, height, angle, force):
    # Create the topic message
	traj = JointTrajectory()
	traj.header = Header()
    # Joint names for UR5
	traj.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                        'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                        'wrist_3_joint']

	rate = rospy.Rate(1)
	pts = JointTrajectoryPoint()
	traj.header.stamp = rospy.Time.now()
	pick_above = [pick_pos[0],pick_pos[1],pick_pos[2]+height, pick_pos[3], pick_pos[4], pick_pos[5]]    #point above target
	place_above = [place_pos[0],place_pos[1],place_pos[2]+height, place_pos[3], place_pos[4], place_pos[5]]   #point above target
	waypoints_pick = build_trajectory(curr_posi, pick_above, seed_state)   #waypoints from current point to 'above pick target'
	waypoints1 = build_trajectory(pick_above, pick_pos, waypoints_pick[-1])   #waypoints from 'above pick target' to 'pick target'
	waypoints_pick.extend(waypoints1)     
	waypoints_place = build_trajectory(pick_above, place_above, waypoints1[0])
	waypoints2 = build_trajectory(place_above, place_pos , waypoints_place[-1])
	waypoints_place.extend(waypoints2)
	while not rospy.is_shutdown():
		for i in range(0,len(waypoints_pick)):
			
			pts.positions = waypoints_pick[i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()             #move to pick

		gripper_control(angle, force)    #grasp it!

		for i in range(0,len(waypoints1)):
			
			pts.positions = waypoints1[len(waypoints1)-1-i]

			pts.time_from_start = rospy.Duration(1.0) 

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()		#go back to pick above

		for i in range(0,len(waypoints_place)):
			pts.positions = waypoints_place[i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()        #go to place 
		
		gripper_control(0, 0)

		for i in range(0,len(waypoints2)):
			
			pts.positions = waypoints2[len(waypoints2)-1-i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()   #go back to place above
		rospy.sleep(1)

		for i in range(0,1):
			pts.positions = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14]
			pts.time_from_start = rospy.Duration(1.0)
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()   #reset arm

pick_pos = [0.2, 0.25, 0.315,-1.57, 0, 0]
place_pos = [-0.205,0.4,0.4,0,0,0]

if __name__ == '__main__':
	try:	
		rospy.init_node('send_joints')
		gripper_cmd_pub = rospy.Publisher(
                	rospy.resolve_name('gripper_controller/gripper_cmd/goal'),
                	GripperCommandActionGoal, queue_size=10)
		pub = rospy.Publisher('/arm_controller/command',
                          JointTrajectory,
                          queue_size=10)
		motionplan(pick_pos,place_pos,0.3,0.4,5)
	except rospy.ROSInterruptException:
		print ("Program interrupted before completion")
