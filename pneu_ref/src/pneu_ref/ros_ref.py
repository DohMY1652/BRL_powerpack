import rospy
from std_msgs.msg import Float32MultiArray

from typing import List
from collections import deque

import numpy as np
import random
import math

from pneu_ref.base_ref import BaseRef
# from base_ref import BaseRef

class ROSRef(BaseRef):
    def __init__(
        self,
        ref_topic_name: str,
        ref_msg_keys: List[str]
    ):
        super(ROSRef, self).__init__()
        
        self.max_time = float('inf')

        self.ref_sub = rospy.Subscriber(ref_topic_name, Float32MultiArray, self.set_goal)
        self.ref_msg_keys = ref_msg_keys

        self.pos_ref = 101.325
        self.neg_ref = 101.325
    
    def set_goal(
        self,
        msg: Float32MultiArray
    ) -> None:
        msg_dict = {}
        for k, v in zip(self.ref_msg_keys, msg.data):
            msg_dict[k] = v
        
        self.pos_ref = msg_dict['pump_ref_pos']
        self.neg_ref = msg_dict['pump_ref_neg']

    def get_goal(
        self,
        curr_time: float, 
    ) -> np.ndarray:
        return np.array([self.pos_ref, self.neg_ref])

if __name__ == '__main__':
    rospy.init_node('reference_tester')
    rospy.loginfo('ROS reference started!')

    ref = ROSRef(
        ref_topic_name = "ref_test",
        ref_msg_keys = ["pump_ref_pos", "pump_ref_neg", "valve_ref_pos", "valve_ref"]
    )

    t = 0
    while not rospy.is_shutdown():
        ref1 = ref.get_goal(t)
        ref2 = ref.get_goal(t + 1)
        ref3 = ref.get_goal(t + 2)
        t += 0.001

        print(t, ref1, ref2, ref3)

    
