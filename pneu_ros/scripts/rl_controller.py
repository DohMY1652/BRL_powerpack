import rospy
from std_msgs.msg import Float32MultiArray

import yaml

from pneu_ref.ros_ref import ROSRef
from pneu_env.env import PneuEnv
from pneu_env.real import PneuReal
from pneu_env.pred import PneuPred
from pneu_rl.sac import SAC
from pneu_ros.ros_bridge import PneuROSBridge
from pneu_utils.utils import get_pkg_path

if __name__ == '__main__':
    with open(f"{get_pkg_path('pneu_ros')}/cfg.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    ros_cfg = cfg['ros']
    controller_cfg = cfg['rl']
    
    rospy.init_node(ros_cfg['node_name'])
    rospy.loginfo('[ INFO] Control Mode: RL')


    model_name = controller_cfg['model_name']
    with open(f'{get_pkg_path("pneu_rl")}/models/{model_name}/cfg.yaml', 'r') as f:
        rl_cfg = yaml.safe_load(f)
    

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


    obs = PneuReal(**controller_cfg["env"]["real"])
    obs.set_init_press(**controller_cfg["env"]["init_press"])
    pred = PneuPred(**controller_cfg["env"]["pred"])

    env = PneuEnv(
        obs = obs,
        ref = ref,
        pred = pred,
        **rl_cfg["env"]
    )
    env.set_pid(**controller_cfg["env"]["pid"])

    model = SAC(
        env = env,
        **rl_cfg["model"]
    )
    model.load(name = controller_cfg["model_name"])

    ctrl_pub = rospy.Publisher(ros_cfg['ctrl']['topic'], Float32MultiArray, queue_size=1)
    
    try:
        state, info = env.reset()

        while not rospy.is_shutdown():
            ctrl_traj = model.predict(state)

            curr_ctrl = ctrl_traj.copy()[:2]
            if controller_cfg["env"]["real"]["scale"]:
                curr_ctrl = 0.3*0.5*(curr_ctrl + 1) + 0.7

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
    

    

        
