#include <ros/ros.h>
#include "catch_bottle.h"




int main(int argc, char** argv)
{
    ros::init(argc, argv, "catch_control");
    ros::NodeHandle nh;

    std::string chain_start, chain_end, urdf_param;
    double timeout;
    nh.param("timeout", timeout, 0.005);
    nh.param("urdf_param", urdf_param, std::string("/robot_description"));
    std::string UR_name_prefix,hand_name_prefix;
    nh.getParam("UR_name_prefix", UR_name_prefix);
    nh.getParam("hand_name_prefix", hand_name_prefix);
    ROS_INFO_STREAM("UR_name_prefix: "<<UR_name_prefix<<"   hand_name_prefix:"<<hand_name_prefix );

    UR UR5e(nh, urdf_param, "world", "robotiq_2f_85_ee_link", timeout, UR_name_prefix,hand_name_prefix);
    ros::Duration(0.5).sleep();
    ros::spinOnce();
//    KDL::JntArray tmp(6);
//    std::vector<double> gravity_up_joints;
//    nh.getParam("gravity_up_joints", gravity_up_joints);
//    for (int j = 0; j <6; ++j) {
//        tmp(j) = gravity_up_joints[j]/180*M_PI;
//    }

//    UR5e.robotiq_hand_move(10,5,100);

//file record path
//    std::string cal_path = "/home/zh/catkin_ws/src/catch_control/config/";
//    UR5e.calibrate_motioncapture_to_UR(cal_path);

//    UR5e.robotiq_hand_move(62,5,100);
//    UR5e.catch_bottle();

//  UR5e.robotiq_hand_move(56,5,50);
    UR5e.spin();

    return  1;


}

