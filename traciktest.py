#!/usr/bin/env python

from trac_ik_python.trac_ik import IK
import rospy
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

#set joint chain
ik_solver = IK("base_link",
               "ee_link")

#set origin position
seed_state = [0,0,0,0,0,0]

#check targer point
print(ik_solver.get_ik(seed_state,0.45, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0))
#check joints
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

	while not rospy.is_shutdown():
		cnt += 1

		if cnt%2 == 1:
			pts.positions = seed_state
		else:
			pts.positions = ik_solver.get_ik(seed_state,
                				0.45, 0.1, 0.3,  # X, Y, Z
                				0.0, 0.0, 0.0, 1.0)  # QX, QY, QZ, QW
		pts.time_from_start = rospy.Duration(1.0)
 		
		traj.points = []
		traj.points.append(pts)
		pub.publish(traj)
		rate.sleep()

if __name__ == '__main__' :
	try:
		main()
	except rospy.ROSInterruptException:
		print("see u next time")





