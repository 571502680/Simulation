#! /usr/bin/env python
#-*- coding: utf-8 -*-
#基础包
import rospy
import actionlib
import numpy as np
import time
#通信接口定义
import ocrtoc_task.msg
from control_msgs.msg import GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
#自己包使用
import robot_control

class CommitSolution(object):
    def __init__(self, name):
        # Init action.
        self.action_name = name
        self.action_server = actionlib.SimpleActionServer(
            self.action_name, ocrtoc_task.msg.CleanAction,
            execute_cb=self.execute_callback, auto_start=False)
        self.action_server.start()
        rospy.loginfo(self.action_name + " is running.")

        self.arm_cmd_pub = rospy.Publisher(
            rospy.resolve_name('arm_controller/command'),
            JointTrajectory, queue_size=10)
        self.gripper_cmd_pub = rospy.Publisher(
            rospy.resolve_name('gripper_controller/gripper_cmd/goal'),
            GripperCommandActionGoal, queue_size=10)

        # create messages that are used to publish feedback/result.
        self.feedback = ocrtoc_task.msg.CleanFeedback()
        self.result = ocrtoc_task.msg.CleanResult()

        # get models directory.
        materials_path = rospy.get_param('~materials_dir',
                                         '/root/ocrtoc_materials')
        self.models_dir = materials_path + '/models'
        rospy.loginfo("Models dir: " + self.models_dir)


    def process_goal_pose(self,goal_pose_list):
        """
        将送入的pose_list变换为可以解析的list
        :param goal_pose_list: goal中的pose_list
        :return:
        """
        return_pose_list=[]
        for pose in goal_pose_list:
            return_pose_list.append(np.array([pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]))
        return return_pose_list

    def execute_callback(self, goal):
        rospy.loginfo("Get clean task.")
        goal_object_list=goal.object_list
        goal_pose_list=goal.pose_list
        goal_pose_list=self.process_goal_pose(goal_pose_list)

        #导入场景
        rospy.loginfo("Load models")
        for object_name in goal.object_list:
            object_model_dir = self.models_dir + '/' + object_name
            rospy.loginfo("Object model dir: " + object_model_dir)
        rospy.sleep(1.0)

        #获取目标场景中位置
        robot=robot_control.Robot()
        objects=robot_control.Objects(get_pose_from_gazebo=False)
        while not rospy.is_shutdown():
            print("Ready to Picking")
            robot.getpose_home(1)
            objects.get_pose()
            robot.home(t=1)
            for i,origin_pose in enumerate(objects.x):
                #1:获取物体名称和物体目标位置
                name=objects.names[i]
                try:
                    goal_index=goal_object_list.index(name)
                    goal_pose=goal_pose_list[goal_index]
                except:
                    print("DenseFusion Don't detect {}".format(name))
                    continue

                #2:抓取物体并运动到目标位置
                print("Traget is {},it's Pose is {}".format(objects.names[i],origin_pose))
                grasp_pose=robot.get_pickpose_from_pose(origin_pose)#Z轴翻转获取物体的抓取Pose
                robot.gripper_control(angle=0,force=1)
                robot.move_updown(grasp_pose,grasp=True,fast_vel=0.4,slow_vel=0.1)

                #运动到目标位置并放下
                robot.home(t=1)
                place_pose=robot.get_pickpose_from_pose(goal_pose)
                #只是用xyz,旋转角度使用grasp的角度
                place_pose[3:]=grasp_pose[3:]
                robot.move_updown(place_pose,grasp=False,fast_vel=0.4,slow_vel=0.1)
            break


        # Example: set status "Aborted" and quit.
        if self.action_server.is_preempt_requested():
            self.result.status = "Aborted"
            self.action_server.set_aborted(self.result)
            return

        # Example: send feedback.
        self.feedback.text = "write_feedback_text_here"
        self.action_server.publish_feedback(self.feedback)
        rospy.loginfo("Pub feedback")
        rospy.sleep(1.0)

        # Example: set status "Finished" and quit.
        self.result.status = "Finished"
        rospy.loginfo("Done.")
        self.action_server.set_succeeded(self.result)
        ##### User code example ends #####


if __name__ == '__main__':
    rospy.init_node('commit_solution')
    commit_solution = CommitSolution('commit_solution')
    rospy.spin()
