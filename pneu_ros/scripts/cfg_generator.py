import rospy

import yaml

from pneu_utils.utils import (
    get_pkg_path,
    color
)
ATM = 101.325

Ku_pos = 0.015
Tu_pos = 2
Ku_neg = 0.011
Tu_neg = 1

cfg = dict(
    ros = dict(
        node_name = "controller",
        ref = dict(
            topic = "/ref_value", 
            msg_keys = [
                "pump_ref_pos", 
                "pump_ref_neg"
            ]),
        sen = dict(
            topic = "/sen_value", 
            msg_keys = [
                "pump_sen_pos", 
                "pump_sen_neg"
            ]),
        ctrl = dict(
            topic = "/RL_pwm", 
            msg_keys = [
                "pump_ctrl_pos", 
                "pump_ctrl_neg"
            ])
    ),
    rl = dict(
        model_name = "Ours000_v09",
        env = dict(
            sim = dict(
                freq = 50,
                delay = 0.1,
                noise = False,
                noise_std = 0,
                offset_pos = 0,
                offset_neg = 0,
                scale = True
            ),
            real = dict(
                freq = 50,
                scale = True
            ),
            pred = dict(
                freq = 50,
                delay = 0,
                noise = False,
                scale = True,
            ),
            init_press = dict(
                init_pos_press = ATM,
                init_neg_press = ATM
            ),
            pid = dict(
                Kp_pos = 0.0,
                Ki_pos = 0.01,
                Kd_pos = 0.0,
                Kp_neg = 0.0,
                Ki_neg = 0.01,
                Kd_neg = 0.0,
                Ka = 10
            )
        )
    ),
    pid = dict(
        freq = 50,
        gains = dict(
            # Kp_pos = Ku_pos,
            # Ki_pos = 0,
            # Kd_pos = 0,
            # Kp_neg = Ku_neg,
            # Ki_neg = 0,
            # Kd_neg = 0,

            # ==> Classic PID <==
            Kp_pos = 0.6*Ku_pos,
            Ki_pos = 1.2*Ku_pos/Tu_pos,
            Kd_pos = 0.075*Ku_pos*Tu_pos,
            Kp_neg = 0.6*Ku_neg,
            Ki_neg = 1.2*Ku_neg/Tu_neg,
            Kd_neg = 0.075*Ku_neg*Tu_neg,

            # # ==> PI <==
            # Kp_pos = 0.45*Ku_pos,
            # Ki_pos = 0.54*Ku_pos/Tu_pos,
            # Kd_pos = 0.0,
            # Kp_neg = 0.45*Ku_neg,
            # Ki_neg = 0.54*Ku_neg/Tu_neg,
            # Kd_neg = 0.0,

            # # ==> Some overshoot <==
            # Kp_pos = Ku_pos/3,
            # Ki_pos = 2/3*Ku_pos/Tu_pos,
            # Kd_pos = 1/9*Ku_pos*Tu_pos,
            # Kp_neg = Ku_neg/3,
            # Ki_neg = 2/3*Ku_neg/Tu_neg,
            # Kd_neg = 1/9*Ku_neg*Tu_neg,

            # # ==> No overshoot <==
            # Kp_pos = 0.2*Ku_pos,
            # Ki_pos = 0.4*Ku_pos/Tu_pos,
            # Kd_pos = 0.06666666*Ku_pos*Tu_pos,
            # Kp_neg = 0.2*Ku_neg,
            # Ki_neg = 0.4*Ku_neg/Tu_neg,
            # Kd_neg = 0.0666666*Ku_neg*Tu_neg,
        ),
        anti_windup = dict(
            Ka = 10
        )
    )
)

# rospy.loginfo("pneu_ros ==> Generating config file starts...")
print("[ INFO] pneu_ros ==> Generating config file starts...")
with open(f"{get_pkg_path('pneu_ros')}/cfg.yaml", "w") as f:
    yaml.dump(cfg, f)
print("[ INFO] pneu_ros ==> Done!")