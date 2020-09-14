#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import math
import numpy as np
from trac_ik_python.trac_ik import IK

pick_above = [0.2451,0.1088,0.2226,-1.5454,-0.0007,1.5645]
start_pos = [0.0002,0.1320,0.6706,-1.5454,-0.0007,1.5645]

def trans_rpy(rpy_info):
	r = rpy_info[3] 
	p = rpy_info[4]
	y = rpy_info[5]
	sinp = math.sin(p/2)
	siny = math.sin(y/2)
	sinr = math.sin(r/2)
	cosp = math.cos(p/2)
	cosy = math.cos(y/2)
	cosr = math.cos(r/2)
	w = round(cosr*cosp*cosy + sinr*sinp*siny,8)
	x = round(sinr*cosp*cosy - cosr*sinp*siny,8)
	y = round(cosr*sinp*cosy + sinr*cosp*siny,8)
	z = round(cosr*cosp*siny - sinr*sinp*cosy,8)

	return [rpy_info[0],rpy_info[1],rpy_info[2],round(x,2),round(y,2),round(z,2),round(w,2)]

Begin = trans_rpy(start_pos)
End = trans_rpy(pick_above)
print(End)
def movetothere(Begin_Pos, End_Pos):

	waypoints_x = [round(Begin_Pos[0],2),round(Begin_Pos[0]+(End_Pos[0]-Begin_Pos[0])*0.4,2),round(Begin_Pos[0]+(End_Pos[0]-Begin_Pos[0])*0.7,2),round(Begin_Pos[0]+(End_Pos[0]-Begin_Pos[0])*1,2)]
	waypoints_y = [round(Begin_Pos[1],2),round(Begin_Pos[1]+(End_Pos[1]-Begin_Pos[1])*0.4,2),round(Begin_Pos[1]+(End_Pos[1]-Begin_Pos[1])*0.7,2),round(Begin_Pos[1]+(End_Pos[1]-Begin_Pos[1])*1,2)]
	waypoints_z = [round(Begin_Pos[2],2),round(Begin_Pos[2]+(End_Pos[2]-Begin_Pos[2])*0.4,2),round(Begin_Pos[2]+(End_Pos[2]-Begin_Pos[2])*0.7,2),round(Begin_Pos[2]+(End_Pos[2]-Begin_Pos[2])*1,2)]
	waypoints_roll=[0,0,0,0,0,0]
	waypoints_pitch = [0,0,0,0,0,0]
	waypoints_yaw = [0,0,0,0,0,0]
	waypoints = map(list, zip(waypoints_x,waypoints_y,waypoints_z,waypoints_roll,waypoints_pitch,waypoints_yaw))
	print(waypoints)
	aim = trans_rpy(waypoints[2])
	print(aim)
	pts.positions = ik_solver.get_ik([1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14], aim[0],aim[1],aim[2],aim[3],aim[4],aim[5],aim[6])
	pts.time_from_start = rospy.Duration(1.0)
	 		
	traj.points = []
	traj.points.append(pts)
	print(traj)
	pub.publish(traj)
	rospy.sleep(3)
	rate.sleep()

ik_solver = IK("base_link",
               "ee_link")
rospy.init_node('send_joints')
pub = rospy.Publisher('/arm_controller/command',JointTrajectory,queue_size=10)
	
traj = JointTrajectory()
traj.header = Header()
	
traj.joint_names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint',
				'wrist_2_joint','wrist_3_joint']
rate = rospy.Rate(10)
#cnt=0
pts = JointTrajectoryPoint()
traj.header.stamp = rospy.Time.now()
def main():

	cnt = 0
	while not rospy.is_shutdown():
		cnt += 1

		if cnt%2 == 1:
	
			movetothere(start_pos,pick_above)

		else:

			rospy.sleep(1)
			rate.sleep()

main()
