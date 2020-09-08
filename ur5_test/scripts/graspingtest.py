#!/usr/bin/env python

from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint
import rospy
import actionlib
from control_msgs.msg import GripperCommandActionGoal
import control_msgs.msg
from trac_ik_python.trac_ik import IK
from control_msgs.msg import GripperCommandActionGoal
from tf.transformations import quaternion_from_euler

ik_solver = IK("base_link", "wrist_3_link")

#seed_state = [0.0] * ik_solver.number_of_joints
seed_state = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14]
pathtopick = [[seed_state[0], seed_state[1], seed_state[2], seed_state[3], seed_state[4], seed_state[5]]]
pathtoplace = [[seed_state[0], seed_state[1], seed_state[2], seed_state[3], seed_state[4], seed_state[5]]]
q = quaternion_from_euler(-1.57, 0, 0)
'''
seed_state = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14]
target = [0.245077,0.108832,0.022651,-1.57,0,1.57]
start_pos = [0.00022,0.13198,0.67058,-1.57,0,1.57]
'''
def gripper_open(m):
	gripper_cmd_pub = rospy.Publisher(
            rospy.resolve_name('gripper_controller/gripper_cmd/goal'),
            GripperCommandActionGoal, queue_size=10)
        gripper_cmd = GripperCommandActionGoal()
        gripper_cmd.goal.command.position = m
        gripper_cmd.goal.command.max_effort = 10
        gripper_cmd_pub.publish(gripper_cmd)
        rospy.loginfo("Pub gripper_cmd")
        rospy.sleep(1.0)



def motionplan(pick_pos, place_pos):
	pick_above = [pick_pos[0],pick_pos[1],pick_pos[2]+0.3]
	place_above = [place_pos[0],place_pos[1],place_pos[2]+0.3]
	waypoints_x = np.linspace(pick_above[0],pick_pos[0],num=5)
	waypoints_y = np.linspace(pick_above[1],pick_pos[1],num=5)
	waypoints_z = np.linspace(pick_above[2],pick_pos[2],num=5)
	waytopick = map(list, zip(waypoints_x,waypoints_y,waypoints_z))
	for i in range(len(waytopick)):
		current_pose = pathtopick[i]
		joint_space = ik_solver.get_ik(current_pose,
                waytopick[i][0], waytopick[i][1],waytopick[i][2],  # X, Y, Z
       	         q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW
		pathtopick.append(joint_space)
	
	waypoints_x = np.linspace(place_above[0],place_pos[0],num=5)
	waypoints_y = np.linspace(place_above[1],place_pos[1],num=5)
	waypoints_z = np.linspace(place_above[2],place_pos[2],num=5)
	waytoplace = map(list, zip(waypoints_x,waypoints_y,waypoints_z))
	for i in range(len(waytoplace)):
		current_pose = pathtoplace[i]
		joint_space = ik_solver.get_ik(current_pose,
                waytoplace[i][0], waytoplace[i][1],waytoplace[i][2],  # X, Y, Z
       	         q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW
		pathtoplace.append(joint_space)

	return pathtopick,pathtoplace
pick_pos = [0.2, 0.25, 0.315]
none = [-0.205,0.4,0.4]
waypoints = motionplan(pick_pos,none)
print(waypoints)


def main():

	rospy.init_node('send_joints')
	pub = rospy.Publisher('/arm_controller/command',
                          JointTrajectory,
                          queue_size=10)
	
	gripper_cmd_pub = rospy.Publisher(
            rospy.resolve_name('gripper_controller/gripper_cmd/goal'),
            GripperCommandActionGoal, queue_size=10)
        gripper_cmd = GripperCommandActionGoal()
	gripper_cmd.goal.command.position = 0.5
	gripper_cmd.goal.command.max_effort = 0.1
	
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

	while not rospy.is_shutdown():
		for i in range(0,len(pathtopick)):
			
			pts.positions = pathtopick[i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()
		rospy.sleep(2)
		gripper_open(0.5)
		for i in range(0,len(pathtopick)):
			m = len(pathtopick)-1
			pts.positions = pathtopick[m-i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()
		rospy.sleep(1)
		for i in range(0,len(pathtoplace)):
			pts.positions = pathtoplace[i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()
		gripper_open(0)
		rospy.sleep(1.0)
		for i in range(0,len(pathtoplace)):
			m = len(pathtopick)-1
			pts.positions = pathtopick[m-i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()
if __name__ == '__main__':
	try:	
		rospy.init_node('send_joints')
		main()
	except rospy.ROSInterruptException:
		print ("Program interrupted before completion")
