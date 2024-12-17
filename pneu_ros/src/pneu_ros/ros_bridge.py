import sys

import rospy
from std_msgs.msg import Float32MultiArray

import json

from typing import List, Dict

from pneu_utils.utils import get_pkg_path

ATM = 101.325

class PneuROSBridge():
    def __init__(
        self,
        sen_topic: str,
        ref_topic: str,
        ctrl_topic: str,
        sen_msg_keys: List[str],
        ref_msg_keys: List[str],
        ctrl_msg_keys: List[str],
    ):
        self.sen_msg_keys = sen_msg_keys
        self.ref_msg_keys = ref_msg_keys
        self.ctrl_msg_keys = ctrl_msg_keys
        
        rospy.Subscriber(sen_topic, Float32MultiArray, self.sen_callback, queue_size=1)
        rospy.Subscriber(ref_topic, Float32MultiArray, self.ref_callback, queue_size=1)
        rospy.Subscriber(ctrl_topic, Float32MultiArray, self.ctrl_callback, queue_size=1)
        
        self.msg = dict(
            time = 0,
            sen_pos = ATM,
            sen_neg = ATM,
            ref_pos = ATM,
            ref_neg = ATM,
            ctrl_pos = 1,
            ctrl_neg = 1
        )

        self.json_file_path = f"{get_pkg_path('pneu_env')}/src/pneu_env/tcpip/obs.json"
    
    def sen_callback(
        self, 
        msg: Float32MultiArray
    ) -> None:
        sen_msg = self.decode_msg(
            msgs = msg.data,
            keys = self.sen_msg_keys
        )
        self.msg['sen_pos'] = sen_msg['pump_sen_pos']
        self.msg['sen_neg'] = sen_msg['pump_sen_neg']
    
    def ref_callback(
        self, 
        msg: Float32MultiArray
    ) -> None:
        ref_msg = self.decode_msg(
            msgs = msg.data,
            keys = self.ref_msg_keys
        )
        self.msg['ref_pos'] = ref_msg['pump_ref_pos']
        self.msg['ref_neg'] = ref_msg['pump_ref_neg']

    def ctrl_callback(
        self, 
        msg: Float32MultiArray
    ) -> None:
        ctrl_msg = self.decode_msg(
            msgs = msg.data,
            keys = self.ctrl_msg_keys
        )

        self.msg['ctrl_pos'] = ctrl_msg['pump_ctrl_pos']
        self.msg['ctrl_neg'] = ctrl_msg['pump_ctrl_neg']

    def decode_msg(
        self,
        msgs: List[float],
        keys: List[str]
    ) -> Dict[str, float]:
        msg_dict = {}
        for k, v in zip(keys, msgs):
            msg_dict[k] = v
        return msg_dict
    
    def write_file(self):
        with open(self.json_file_path, 'w') as f:
            json.dump(self.msg, f)


if __name__ == "__main__":
    rospy.init_node("bridge")
    bridge = PneuROSBridge(
        sen_topic = "sen_value",
        ref_topic = "ref_value", 
        ctrl_topic = "RL_pwm",
        sen_msg_keys = ["pump_sen_pos", "pump_sen_neg"],
        ref_msg_keys = ["pump_ref_pos", "pump_ref_neg"],
        ctrl_msg_keys = ["pump_ctrl_pos", "pump_ctrl_neg"]
    )

    rospy.spin()
    