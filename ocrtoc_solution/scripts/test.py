# %%
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

# ur = rc.Robot( base_link ="world", tip_link = "robotiq_2f_85_ee_link" )
ur = rc.Robot( base_link ="base_link", tip_link = "wrist_3_link" )
obj = rc.objects()
# ik_solver.
# urdf_str = rospy.get_param('/robot_description')
# print(urdf_str)
# %%

# %%
rospy.sleep(0.5)
# ur.move_to_joint([1.5621, -2.12, 1.72, -1.1455, -1.57, 0], 10)

# ur.home()
print(ur.q)
print(ur.x)
print(ur.p)
# ur.sin_test()
# rospy.sleep(1.0)
# print(ur._num_jnts)

# %%

# print(qd)
# ur.move_to_frame(qd,5)
# print(ur.x)



# %%












