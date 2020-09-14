
#! /usr/bin/env python
import rospy
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import numpy as np

import robot_control as rc
from trac_ik_python.trac_ik import IK

# import PyKDL as kdl
# %%
## - 
rospy.init_node('test_motion_control')

ur = rc.Robot( base_link ="table", tip_link = "robotiq_2f_85_ee_link" )
# ur = rc.Robot( base_link ="base_link", tip_link = "robotiq_2f_85_ee_link" )
obj = rc.objects()
# ik_solver.
# urdf_str = rospy.get_param('/robot_description')
# print(urdf_str)
# %%

# %%
rospy.sleep(0.5)
ur.home(t=4)


print ur._joint_names

print(ur.q)
print(ur.p)
# ur.sin_test()
# rospy.sleep(1.0)
# print(ur._num_jnts)

# %%
print obj.x[1]



# %%












