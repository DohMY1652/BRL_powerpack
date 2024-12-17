import rospy
from std_msgs.msg import Float32MultiArray

import yaml

from pneu_ref.ros_ref import ROSRef
from pneu_env.env import PneuEnv
from pneu_env.real import PneuReal
from pneu_env.pred import PneuPred
from pneu_pid.pid import PID
from pneu_ros.ros_bridge import PneuROSBridge
from pneu_utils.utils import get_pkg_path

if __name__ == '__main__':
    with open(f"{get_pkg_path('pneu_ros')}/cfg.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    ros_cfg = cfg['ros']
    controller_cfg = cfg['pid']
    
    rospy.init_node(ros_cfg['node_name'])
    rospy.loginfo('[ INFO] Control Mode: RL')

    ref = ROSRef(
        ref_topic_name = ros_cfg['ref']['topic'],
        ref_msg_keys = ros_cfg['ref']['msg_keys']
    )

    bridge = PneuROSBridge(
        sen_topic = ros_cfg['sen']['topic'],
        ref_topic = ros_cfg['ref']['topic'],
        ctrl_topic = ros_cfg['ctrl']['topic'],
        sen_msg_keys = ros_cfg['sen']['msg_keys'],
        ref_msg_keys = ros_cfg['ref']['msg_keys'],
        ctrl_msg_keys = ros_cfg['ctrl']['msg_keys']
    )


    obs = PneuReal(freq=controller_cfg["freq"], scale=False)
    # obs.set_init_press(**controller_cfg["env"]["init_press"])
    # pred = PneuPred(**controller_cfg["env"]["pred"])

    env = PneuEnv(
        obs = obs,
        ref = ref,
        num_prev = 1,
        num_act = 1,
        num_pred = 1
    )

    model = PID(env, **controller_cfg['gains'])
    model.set_anti_windup(**controller_cfg["anti_windup"])

    ctrl_pub = rospy.Publisher(ros_cfg['ctrl']['topic'], Float32MultiArray, queue_size=1)
    
    try:
        state, info = env.reset()

        while not rospy.is_shutdown():
            ctrl_traj = model.predict(state)

            curr_ctrl = ctrl_traj.copy()[:2]

            msg = Float32MultiArray()
            msg.data = list(curr_ctrl)
            ctrl_pub.publish(msg)

            bridge.write_file()
            state, _, _, _, info = env.step(ctrl_traj)

    except Exception as e:
        rospy.logerr(f"Error: {e}")

    finally:
        env.close()
        rospy.loginfo("[INFO] Environment closed.")
    

    

        
