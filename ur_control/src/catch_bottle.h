//
// Created by xg on 18-11-3.
//

#ifndef PROJECT_CATCH_BOTTLE_H
#define PROJECT_CATCH_BOTTLE_H


#include <ros/ros.h>
#include <trac_ik/trac_ik.hpp>
#include<sensor_msgs/JointState.h>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryGoal.h>
#include <std_msgs/Float64MultiArray.h>
#include <fstream>
#include <iostream>
#include <kdl_parser/kdl_parser.hpp>
//#include "robotiq_85_msgs/GripperCmd.h"
//#include "robotiq_85_msgs/GripperStat.h"

//#include <Eigen/Geometry>
//#include <Eigen/Core>


class UR{

    typedef  actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>  Client;

public:
    UR(ros::NodeHandle _nh, const std::string & _urdf_param, const std::string & _chain_start, const std::string & _chain_end, double _timeout,std::string UR_name_prefix, std::string hand_name_prefix){
        double eps = 1e-5;
        nh_ = _nh;
        p_tracik_solver_ = new TRAC_IK::TRAC_IK(_chain_start, _chain_end, _urdf_param, _timeout, eps); //反解
        KDL::Chain chain;
        bool valid = p_tracik_solver_->getKDLChain(chain);
        if (!valid) {
            ROS_ERROR("There were no valid KDL chain found");

        }

        KDL::Chain chain2;
        KDL::Tree tree;
        std::string robot_desc_string;
        nh_.param("robot_description", robot_desc_string, std::string("robot_description"));
        if (kdl_parser::treeFromString(robot_desc_string, tree)) {
            std::string base_link =  _chain_start;
            std::string tip_link =  _chain_end;
            if (tree.getChain(_chain_start, _chain_end, chain2)) {
                std::vector<KDL::Segment> segments = chain2.segments;
                for (int x = 0; x < segments.size(); ++x) {
                    KDL::Segment seg = segments.at(x);
                    std::string name = seg.getName();
                    KDL::Joint jnt = seg.getJoint();

                    KDL::Vector axis = jnt.JointAxis();
                    KDL::Frame frame = seg.getFrameToTip();
                    double r, p, y;
                    frame.M.GetRPY(r, p, y);
                    //output
                    ROS_INFO_STREAM("KDL:: name=" << name << "; "
                                                  << "axis=[" << axis[0] << "," << axis[1] << "," << axis[2] << "]; "
                                                  << "frame.p=[" << frame.p[0] << "," << frame.p[1] << "," << frame.p[2]
                                                  << "]; "
                                                  << "frame.rpy=[" << r << "," << p << "," << y << "]; "
                    );
                }
                p_fk_solver_ = new KDL::ChainFkSolverPos_recursive(chain2);
//                p_jac_solver_ = new KDL::ChainJntToJacSolver(chain2);
            } else {
                ROS_FATAL("Couldn't find chain %s to %s", _chain_start.c_str(), _chain_end.c_str());
            }
            //ik
            double timeout;
            nh_.param("trac_ik_timeout", timeout, 0.005);
            
            p_tracik_solver_ = new TRAC_IK::TRAC_IK(base_link, tip_link, _urdf_param, timeout, eps);
        } else {
            ROS_FATAL("Failed to extract kdl tree from xml robot description");
        }

//        p_fk_solver_ = new KDL::ChainFkSolverPos_recursive(chain2);  //正解
        ROS_INFO ("Using %d joints",chain2.getNrOfJoints());
        ROS_INFO ("segment: %d",chain2.getNrOfSegments());
        //init
        joint_names_.push_back("shoulder_pan_joint");
        joint_names_.push_back("shoulder_lift_joint");
        joint_names_.push_back("elbow_joint");
        joint_names_.push_back("wrist_1_joint");
        joint_names_.push_back("wrist_2_joint");
        joint_names_.push_back("wrist_3_joint");
        current_JntArr_.resize(6);
        for(int i=0;i<6;++i)
        {
            if (i==1||i==3)
                current_JntArr_(i) = -M_PI/2;
            else if(i==0)
                current_JntArr_(i) = M_PI;
            else
                current_JntArr_(i) = 0;
        }
	p_fk_solver_->JntToCart(current_JntArr_, frame_wrist3_base_); 
	ROS_INFO_STREAM("current pos: x="<<frame_wrist3_base_.p.data[0]<<"  y="<<frame_wrist3_base_.p.data[1]<<"  z="<<frame_wrist3_base_.p.data[2]);
        bsub_ = false;
        bstart_cal_ = false;

        sub_=nh_.subscribe("/joint_states", 1, &UR::subJointStatesCB, this);
        sub_command_ =  nh_.subscribe("/ur_command", 1, &UR::subCommandCB, this);
//        sub_hand_ = nh_.subscribe("gripper/stat",1 , &UR::subHandStatesCB,this);
//        pub_hand_cmd_ = nh_.advertise<robotiq_85_msgs::GripperCmd>("gripper/cmd",1);
        pub_end_pose_ = nh_.advertise<std_msgs::Float64MultiArray>("ur_pose",1);


        ROS_INFO("sub done");

        //action client
        client_ = new Client("/arm_controller/follow_joint_trajectory", true);
        client_->waitForServer(ros::Duration());
  
    }

    ~UR(){
        delete p_tracik_solver_;
	delete p_fk_solver_;
        delete client_;

    }

    void spin()
    {	

        ros::spin();
	// ros::Rate rate(125);
	// while(ros::ok()){
	// ros::spinOnce();
	// p_fk_solver_->JntToCart(current_JntArr_, frame_wrist3_base_);    //frame_wrist3_base_ 为正运动学计算出的位姿


    //     // ROS_INFO_STREAM("current pos: x="<<frame_wrist3_base_.p.data[0]<<"  y="<<frame_wrist3_base_.p.data[1]<<"  z="<<frame_wrist3_base_.p.data[2]);
    //    // ROS_INFO_STREAM("current j0 ="<<current_JntArr_(0)<<"  j1="<<current_JntArr_(1)<<"  j2="<<current_JntArr_(2));
       
	
	// std_msgs::Float64MultiArray end_pose;
    //     end_pose.data.resize(7);
    //     end_pose.data[0] = frame_wrist3_base_.p.data[0];
    //     end_pose.data[1] = frame_wrist3_base_.p.data[1];
    //     end_pose.data[2] = frame_wrist3_base_.p.data[2];
    //     double qx=0;
    //     double qy=0;
    //     double qz=0;
    //     double qw=0;

    //     frame_wrist3_base_.M.GetQuaternion(qx,qy,qz,qw);
    //     end_pose.data[3] = qx;
    //     end_pose.data[4] = qy;
    //     end_pose.data[5] = qz;
    //     end_pose.data[6] = qw;

    //     pub_end_pose_.publish(end_pose);
	
	// rate.sleep();
	// }
        


    }

    void moveto_joints(KDL::JntArray target_jnt, double time){
//        ROS_INFO("move to joints");
        //运行前需要ros::spinOnce 来更新当前关节角
        bur_sub_ = false;
        do{
            ros::spinOnce();
        }while(!bur_sub_);

        trajectory_msgs::JointTrajectoryPoint p0;
        trajectory_msgs::JointTrajectoryPoint p1;
        control_msgs::FollowJointTrajectoryGoal g;
        g.trajectory.header.stamp = ros::Time::now();
        g.trajectory.joint_names.push_back("shoulder_pan_joint");
        g.trajectory.joint_names.push_back("shoulder_lift_joint");
        g.trajectory.joint_names.push_back("elbow_joint");
        g.trajectory.joint_names.push_back("wrist_1_joint");
        g.trajectory.joint_names.push_back("wrist_2_joint");
        g.trajectory.joint_names.push_back("wrist_3_joint");

        for(int x=0;x<6;++x)
        {
            p0.positions.push_back(current_JntArr_(x));
            p0.velocities.push_back(0);
        }
        p0.time_from_start = ros::Duration(0);
        g.trajectory.points.push_back(p0);

        for(int x=0;x<6;++x)
        {
            p1.positions.push_back(target_jnt(x));
            p1.velocities.push_back(0);
        }
        p1.time_from_start = ros::Duration(time);
        g.trajectory.points.push_back(p1);
        client_->sendGoal(g);
        client_->waitForResult(ros::Duration());

    }




    void force_update_all_state(){
//        b_hand_sub_ = false;
        bur_sub_ = false;

        do{
            ros::spinOnce();
        }while(!bur_sub_ );
    }

    void move_to_xyz(double x,double y, double z,double time,bool b_relative){
        force_update_all_state();
        KDL::Frame next_frame = frame_wrist3_base_;
        KDL::JntArray next_jnts(6);
        if(b_relative){
            ROS_INFO_STREAM("move to position relative to current point   x="<<x<<"   y="<<y<<"  z"<<z);
            next_frame.p.data[0] += x;
            next_frame.p.data[1] += y;
            next_frame.p.data[2] += z;

        } else{
            ROS_INFO_STREAM("move to fixed position     x="<<x<<"   y="<<y<<"  z"<<z);
            next_frame.p.data[0] = x;
            next_frame.p.data[1] = y;
            next_frame.p.data[2] = z;
        }
        p_tracik_solver_->CartToJnt(current_JntArr_,next_frame,next_jnts);
        moveto_joints(next_jnts,time);
    }




    void subCommandCB(std_msgs::Float64MultiArray state)
    {
//     state: [x,y,z,,qx,qy,qz,qw,time]
        KDL::Frame next_frame = frame_wrist3_base_;
        KDL::JntArray next_jnts(6);
        next_frame.p.data[0] = state.data[0];
        next_frame.p.data[1] = state.data[1];
        next_frame.p.data[2] = state.data[2];
        next_frame.M = KDL::Rotation::Quaternion(state.data[3],
                                                 state.data[4],
                                                 state.data[5],
                                                 state.data[6]);
        p_tracik_solver_->CartToJnt(current_JntArr_,next_frame,next_jnts);
        double exe_time = state.data[7];
        ROS_INFO_STREAM("receive command: x="<<state.data[0]<<"  y="<<state.data[0]<<"  z="<<state.data[0]<<"  t="<<state.data[7]);
        moveto_joints(next_jnts, exe_time);


    }

    void subJointStatesCB(sensor_msgs::JointState state)
    {
        KDL::JntArray jntArr;
        KDL::JntArray jntSpeed;
        KDL::JntArray jntCur;
        jntArr.resize(6);
        jntSpeed.resize(6);
        jntCur.resize(6);
        int n = state.name.size();
        for (int i = 0; i < 6; ++i)//joint_names_
        {
            int x = 0;
            for (; x < n; ++x)//state
            {
                if (state.name[x] == ( joint_names_[i])) {
                    jntArr(i) = state.position[x];
                    jntSpeed(i) = state.velocity[x];
                    jntCur(i) = state.effort[x];
                    break;
                }
            }

            if (x == n) {
                ROS_ERROR_STREAM("Error,  joint name : " << joint_names_[i] << " , not found.  ");
                return;
            }
        }
        current_JntArr_ = jntArr;
//      current_JntSpeed = jntSpeed;
//      current_JntCur_ = jntCur;

        bur_sub_ = true;
        
    }

private:
    ros::NodeHandle nh_;
    TRAC_IK::TRAC_IK *p_tracik_solver_;
    KDL::ChainFkSolverPos_recursive *p_fk_solver_;
    ros::Subscriber sub_;   //UR5e sub
    ros::Subscriber sub_command_;   //UR5e control command  sub
    KDL::JntArray current_JntArr_;
    Client *client_;
    Client *client_servoj_;
    bool bsub_;
    bool bur_sub_;
    std::vector<std::string> joint_names_;
    KDL::Frame frame_wrist3_base_;//fk result
    KDL::Vector end_point_;

    // Robotiq Hand
    ros::Subscriber sub_hand_;
    ros::Publisher pub_hand_cmd_;
    ros::Publisher pub_end_pose_;

    //Motion capture
    ros::Subscriber sub_markers_;
    bool bmotion_capture_sub_;
    bool bstart_cal_;
    std::vector<KDL::Vector> first_body_markers_ ;



    bool b_hand_sub_;
};








#endif //PROJECT_CATCH_BOTTLE_H

