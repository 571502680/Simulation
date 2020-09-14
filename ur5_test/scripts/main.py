#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory

from trajectory_msgs.msg import JointTrajectoryPoint

waypoints = [[1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14],[0,0,0,0,0,0]]

def main():
	
	rospy.init_node('send_joints')
	pub = rospy.Publisher('/arm_controller/command',JointTrajectory,queue_size=10)
	
	traj = JointTrajectory()
	traj.header = Header()
	
	traj.joint_names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint',
				'wrist_2_joint','wrist_3_joint']
	rate = rospy.Rate(1)
	cnt=0
	pts = JointTrajectoryPoint()
	traj.header.stamp = rospy.Time.now()

	while not rospy.is_shutdown():
		cnt += 1

		if cnt%2 == 1:
			pts.positions = waypoints[0]
		else:
			pts.positions = waypoints[1]
		pts.time_from_start = rospy.Duration(1.0)
 		
		traj.points = []
		traj.points.append(pts)
		pub.publish(traj)
		rospy.sleep(3)
		rate.sleep()

if __name__ == '__main__' :
	try:
		main()
	except rospy.ROSInterruptException:
		print("you are too lame")
