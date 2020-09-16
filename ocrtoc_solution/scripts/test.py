
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

ur = rc.Robot( base_link ="world", tip_link = "robotiq_2f_85_ee_link" )
# ur = rc.Robot( base_link ="base_link", tip_link = "robotiq_2f_85_ee_link" )
obj = rc.objects()
# ik_solver.
# urdf_str = rospy.get_param('/robot_description')
# print(urdf_str)
# %%

# %%
rospy.sleep(1)
ur.home(t=4)
rospy.sleep(4)

p1,p2,p3,p4 = np.copy(ur.x),np.copy(ur.x),np.copy(ur.x),np.copy(ur.x)
print ur.x
offset = 0.3
p1[:3] = p1[:3] + np.array([0, offset, 0  ]  )
p2[:3] = p2[:3] + np.array([offset, offset, 0  ]  )
p3[:3] = p3[:3] + np.array([offset, -offset, 0  ]  )
p4[:3] = p4[:3] + np.array([offset, -offset -0.1, 0  ]  )

poses = np.concatenate((p1.reshape(1,-1),p2.reshape(1,-1),p3.reshape(1,-1),p4.reshape(1,-1) ))
# print p1
ur.motion_generation(poses, vel=0.2)

ur.home(t=4)
# print range(1,4)
# print(ur.q)
# print(ur.p)
# ur.sin_test()
# rospy.sleep(1.0)
# print(ur._num_jnts)

# %%
# print obj.x[1]



# %%












