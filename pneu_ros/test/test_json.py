import rospy
from std_msgs.msg import Float32MultiArray

import json

from typing import List, Dict

from pneu_utils.utils import get_pkg_path

rospy.init_node("json_test_node")

json_ctrl_file = f"{get_pkg_path('pneu_env')}/src/pneu_env/tcpip/ctrl.json"
json_obs_file = f"{get_pkg_path('pneu_env')}/src/pneu_env/tcpip/obs.json"

rate = rospy.Rate(100)

while not rospy.is_shutdown():
    with open(json_ctrl_file, 'r') as f:
        d = json.load(f)
    print(d)
    rate.sleep()


    