#!/usr/bin/env python

from trac_ik_python.trac_ik import IK
import rospy
import math
import time
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

def EulerAndQuaternionTransform( intput_data):
	data_len = len(intput_data)
	angle_is_not_rad = False
 
	if data_len == 3:
		r = 0
		p = 0
		y = 0
		if angle_is_not_rad: # 180 ->pi
			r = math.radians(intput_data[0]) 
			p = math.radians(intput_data[1])
			y = math.radians(intput_data[2])
		else:
			r = intput_data[0] 
			p = intput_data[1]
			y = intput_data[2]
 
		sinp = math.sin(p/2)
		siny = math.sin(y/2)
		sinr = math.sin(r/2)
 
		cosp = math.cos(p/2)
		cosy = math.cos(y/2)
		cosr = math.cos(r/2)
 
		w = cosr*cosp*cosy + sinr*sinp*siny
		x = sinr*cosp*cosy - cosr*sinp*siny
		y = cosr*sinp*cosy + sinr*cosp*siny
		z = cosr*cosp*siny - sinr*sinp*cosy
 
		return [w,x,y,z]
 
	elif data_len == 4:
 
		w = intput_data[0] 
		x = intput_data[1]
		y = intput_data[2]
		z = intput_data[2]
 
		r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
		p = math.asin(2 * (w * y - z * x))
		y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
 
		if angle_is_not_rad ==False: # 180 ->pi
 
			r = r / math.pi * 180
			p = p / math.pi * 180
			y = y / math.pi * 180
 
		return [r,p,y]


box3=EulerAndQuaternionTransform([-0.0,0.0,1.012175])
target=[0.1580,-0.1388,0.015]+box3
print(target)
#set the joint chain
ik_solver = IK("base_link",
               "ee_link")

#set the origin state
seed_state = [0,0,0,0,0,0]
#print(ik_solver.get_ik(seed_state,0.45, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0))
#joint_names=list(ik_solver.joint_names()
print(ik_solver.joint_names)
def main():	
	rospy.init_node('send_joints')
	pub = rospy.Publisher('/arm_controller/command',JointTrajectory,queue_size=10)

	traj = JointTrajectory()
	traj.header = Header()
	traj.joint_names = list(ik_solver.joint_names)
		#traj.joint_names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint',
					#'wrist_2_joint','wrist_3_joint']
	rate = rospy.Rate(1)
	cnt=0
	pts = JointTrajectoryPoint()
	traj.header.stamp = rospy.Time.now()

	pts.positions = seed_state
	traj.points = []
	traj.points.append(pts)
	pub.publish(traj)

	time.sleep(3)
	pts.positions = ik_solver.get_ik(seed_state,0.15, -0.2, 0.35     ,  # X, Y, Z
	                		0.0, 0.0, 0.0, 1.0)  # QX, QY, QZ, QW
	pts.time_from_start = rospy.Duration(1.0)
	#pts.positions = ik_solver.get_ik(seed_state,target[0], target[1],target[2]+0.3,target[3],target[4],target[5],target[6])		
	traj.points = []
	traj.points.append(pts)
	pub.publish(traj)
	rate.sleep()

if __name__ == '__main__' :
	try:
		main()
	except rospy.ROSInterruptException:
		print("over")



