#!/usr/bin/env python

from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint
import rospy
import actionlib
import control_msgs.msg
from trac_ik_python.trac_ik import IK

from tf.transformations import quaternion_from_euler

ik_solver = IK("base_link", "wrist_3_link")

#seed_state = [0.0] * ik_solver.number_of_joints
seed_state = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14]
waypoints = [[seed_state[0], seed_state[1], seed_state[2], seed_state[3], seed_state[4], seed_state[5]]]
waypoints2 = [[seed_state[0], seed_state[1], seed_state[2], seed_state[3], seed_state[4], seed_state[5]]]
q = quaternion_from_euler(-1.57, 0, -1.57)
'''
seed_state = [1.5621, -2.12, 1.72, -1.1455, -1.57, 3.14]
target = [0.245077,0.108832,0.022651,-1.57,0,1.57]
start_pos = [0.00022,0.13198,0.67058,-1.57,0,1.57]
'''
def motionplan(pick_pos, place_pos):
	pick_above = [pick_pos[0],pick_pos[1],pick_pos[2]+0.3]
	place_above = [place_pos[0],place_pos[1],place_pos[2]+0.3]
	waypoints_x = np.linspace(pick_above[0],pick_pos[0],num=3)
	waypoints_y = np.linspace(pick_above[1],pick_pos[1],num=3)
	waypoints_z = np.linspace(pick_above[2],pick_pos[2],num=3)
	waytopick = map(list, zip(waypoints_x,waypoints_y,waypoints_z))
	for i in range(len(waytopick)):
		current_pose = waypoints[i]
		joint_space = ik_solver.get_ik(current_pose,
                waytopick[i][0], waytopick[i][1],waytopick[i][2],  # X, Y, Z
       	         q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW
		waypoints.append(joint_space)
	goback = []
	for i in waypoints:
		goback.append(i)
	goback.reverse()
	waypoints.extend(goback)
	waypoints_x = np.linspace(place_above[0],place_pos[0],num=3)
	waypoints_y = np.linspace(place_above[1],place_pos[1],num=3)
	waypoints_z = np.linspace(place_above[2],place_pos[2],num=3)
	waytoplace = map(list, zip(waypoints_x,waypoints_y,waypoints_z))
	for i in range(len(waytoplace)):
		current_pose = waypoints2[i]
		joint_space = ik_solver.get_ik(current_pose,
                waytoplace[i][0], waytoplace[i][1],waytoplace[i][2],  # X, Y, Z
       	         q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW
		waypoints2.append(joint_space)
	goback = []
	for i in waypoints2:
		goback.append(i)
	goback.reverse()
	waypoints2.extend(goback)
	waypoints.extend(waypoints2)
	return waypoints
pick_pos = [0.3, 0.32, 0.3]
none = [-0.2,0.4,0.4]
waypoints = motionplan(pick_pos,none)
print(waypoints)
'''
waytogo = movetothere(start_pos,target)

q = quaternion_from_euler(waytogo[0][3], waytogo[0][4], waytogo[0][5])

joint_space = ik_solver.get_ik(seed_state,
                waytogo[0][0], waytogo[0][1],waytogo[0][2],  # X, Y, Z
       	         q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW
current_joint = [joint_space[0], joint_space[1], joint_space[2], joint_space[3], joint_space[4], joint_space[5]]
waypoints = [[joint_space[0], joint_space[1], joint_space[2], joint_space[3], joint_space[4], joint_space[5]]]

q = quaternion_from_euler(waytogo[4][3], waytogo[4][4], waytogo[4][5])

joint_space = ik_solver.get_ik(current_joint,
                waytogo[4][0], waytogo[4][1],waytogo[4][2],  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

#print(joint_space)
waypoints.append(joint_space)


# Convert RPY to Quaternions
q = quaternion_from_euler(-1.57, 0, -1.57)
#q = quaternion_from_euler(0, 0, 0)
joint_space = ik_solver.get_ik(seed_state,
                0.15, 0.32,0.45,  # X, Y, Z
       	         q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

#print(joint_space)
waypoints = [[joint_space[0], joint_space[1], joint_space[2], joint_space[3], joint_space[4], joint_space[5]]]

joint_space = ik_solver.get_ik(waypoints[0],
                0.15, 0.32, 0.3,  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW
waypoints.append(joint_space)

joint_space = ik_solver.get_ik(waypoints[1],
                0.15, 0.32, 0.2,  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

waypoints.append(joint_space)

joint_space = ik_solver.get_ik(waypoints[2],
                0.25, 0.32, 0.4,  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

waypoints.append(joint_space)

def gripper_control(m):
	rospy.init_node('gripper_control')

# Create an action client
	client = actionlib.SimpleActionClient(
   		'/gripper_controller/gripper_cmd',  # namespace of the action topics
    		control_msgs.msg.GripperCommandAction )# action type
	rate = rospy.Rate(1)  	
# Wait until the action server has been started and is listening for goals
	client.wait_for_server()
 
# Create a goal to send (to the action server)
	goal = control_msgs.msg.GripperCommandGoal()
 	goal.command.position = m
	#goal.command.position = value   # From 0.0 to 0.8
	goal.command.max_effort = -1.0  # Do not limit the effort
	client.send_goal(goal)
	client.wait_for_result()

# Convert RPY to Quaternions
q = quaternion_from_euler(0, 0, 0)

joint_space = ik_solver.get_ik(seed_state,
                0.45, 0.50, 0.40,  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

#print(joint_space)
waypoints.append(joint_space)

# Convert RPY to Quaternions
q = quaternion_from_euler(0, 0, 0)

joint_space = ik_solver.get_ik(seed_state,
                0.45, 0.40, 0.4,  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

#print(joint_space)
waypoints.append(joint_space)

# Convert RPY to Quaternions
q = quaternion_from_euler(0, 0, 0)

joint_space = ik_solver.get_ik(seed_state,
                0.45, 0.40, 0.3,  # X, Y, Z
       	        q[0], q[1], q[2], q[3])  # QX, QY, QZ, QW

#print(joint_space)
waypoints.append(joint_space)
'''
def main():

	rospy.init_node('send_joints')
	pub = rospy.Publisher('/arm_controller/command',
                          JointTrajectory,
                          queue_size=10)

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
		for i in range(0,len(waypoints)):
			
			pts.positions = waypoints[i]

			pts.time_from_start = rospy.Duration(1.0)

            # Set the points to the trajectory
			traj.points = []
			traj.points.append(pts)
            # Publish the message
			pub.publish(traj)
			rate.sleep()

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		print ("Program interrupted before completion")
