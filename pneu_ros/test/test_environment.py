import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray

from pneu_env.sim import PneuSim


class PneuROSEnvTester():
    def __init__(self, env):
        self.env = env

        self.sen_pub = rospy.Publisher('/sen_value', Float32MultiArray, queue_size=1)

        rospy.Subscriber('/ref_value', Float32MultiArray, self.ref_callback)
        rospy.Subscriber('/RL_pwm', Float32MultiArray, self.ctrl_callback)

        self.ref_keys = ["pump_ref_pos", "pump_ref_neg"]
        self.ctrl_keys = ["pump_ctrl_pos", "pump_ctrl_neg"]

        self.curr_ref = np.array([101.325, 101.325])
        self.curr_ctrl = np.array([1, 1])
        
    def ctrl_callback(self, msg):
        ctrl = self.decode_msg(msg.data, self.ctrl_keys)
        self.curr_ctrl = np.array([ctrl["pump_ctrl_pos"], ctrl["pump_ctrl_neg"]])

    def ref_callback(self, msg):
        ref = self.decode_msg(msg.data, self.ref_keys)
        self.curr_ref = np.array([ref["pump_ref_pos"], ref["pump_ref_neg"]])
    
    def publish_sen(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            next_obs, obs_info = self.env.observe(self.curr_ctrl, self.curr_ref)
            rospy.loginfo(f"Observation : {list(next_obs)}")

            msg = Float32MultiArray()
            msg.data = list(next_obs[1:])
            self.sen_pub.publish(msg)

            rate.sleep()
            

    def decode_msg(self, msg, key):
        m = {}
        for k, v in zip(key, msg):
            m[k] = v
        return m


if __name__ == "__main__":
    rospy.init_node('pneu_env_test_node')
    rospy.loginfo("Tester environment is initiated")

    sim = PneuSim(
        freq = 50,
        delay = 0.1,
        scale = False
    )

    tester = PneuROSEnvTester(sim)
    tester.publish_sen()
    