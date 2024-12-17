import sys
import rospy

from ctypes import CDLL, c_double, POINTER

import numpy as np
import math
from typing import Tuple, Dict, Any, Union, List, Optional
from collections import deque

from gymnasium.spaces import Box

from pneu_env.sim import PneuSim
from pneu_env.real import PneuReal
from pneu_msg.msg import Observation

from pneu_utils.utils import get_pkg_path, checker
from pneu_ref.pneu_ref import PneuRef

class PneuEnv():
    def __init__(
        self,
        obs: Any,
        ref: Any,
        pred_obs: Optional[PneuSim] = None,
        num_obs: int = 10,
        num_ref: int = 10,
        enable_obs_pub: bool = False,
    ):
        self.obs = obs
        self.goal = PneuRef(
            ref,
            num_ref = num_ref
        )
        self.obs_pred = pred_obs
        
        self.num_obs = num_obs
        self.num_ref = num_ref
        
        self.action_space = Box(
            low = -1.,
            high = 1.,
            shape = (2,),
            dtype = np.float64
        )
        
        self.dim_ref, self.dim_obs = 2, 2
        self.dim_ref_set = self.dim_ref*num_ref
        self.dim_obs_set = self.dim_obs*num_obs
        state_dim = self.dim_ref_set + self.dim_obs_set
        self.observation_space = Box(
            low = -np.inf,
            high = np.inf,
            shape = (state_dim, ),
            dtype = np.float64
        )
        
        self.obs_buf = 101.325*np.ones(self.dim_obs_set, dtype=np.float64)
        obs_idxs = deque()
        for i in range(num_obs):
            obs_idxs.append(2*i) 
            obs_idxs.append(2*i + 1) 
        self.obs_idxs = np.array(obs_idxs)
        self.time_buf = 0.0

        self.obs_pub_flag = enable_obs_pub
        if enable_obs_pub:
            rospy.init_node('pneu_env', anonymous=True)
            self.obs_pub = rospy.Publisher('/obs', Observation, queue_size=1)
            self.obs_wo_noise_pub = rospy.Publisher('/obs_wo_noise', Observation, queue_size=1)
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl = np.array([1.0, 1.0], dtype=np.float64)
        state, _, _, _, info = self.step(ctrl)
        self.time_buf = info['obs']['curr_time']
        
        return state, info 
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        next_state, state_info = self.get_state(action)
        reward, reward_info = self.get_reward(
            state=next_state,
            action = action
        )
        terminated = False
        truncated = False

        info = self.get_info(state_info, reward_info) 
        
        self.verbose(info)
        if self.obs_pub_flag is True:
            self.publish_observation(info)

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )
        
    def close(self):
        ctrl = np.array([1.0, 1.0], dtype=np.float64)
        goal = np.array([101.325, 101.325], dtype=np.float64)
        _, _ = self.obs.get_obs(ctrl, goal)

    def get_state(
        self,
        ctrl: np.ndarray
    ) -> np.ndarray:
        goal = self.goal.get_ref(self.time_buf)
        obs, obs_info = self.obs.get_obs(ctrl, goal.copy()[0:2])
        self.obs_buf = np.concatenate((obs[1:3], self.obs_buf[:-2]), axis=0)

        self.time_buf = obs[0]

        obs_set = self.obs_buf[self.obs_idxs]
        state = np.concatenate((goal, obs_set), axis=0)

        obs_info['ctrl'] = ctrl

        return state, obs_info
    
    def predict_obs(
        self,
        init_pos_press: float,
        init_neg_press: float,
        ctrl_traj: np.ndarray
    ) -> np.ndarray:
        self.obs_pred.set_init_press(
            init_pos_press,
            init_neg_press
        )

        obses = np.array([init_pos_press, init_neg_press])
        for ctrl in ctrl_traj.reshape(-1,2):
            obs, _ = self.get_obs(
                ctrl = 2*ctrl - 1,
                # ctrl = [0, 0],
                goal = np.array([101.325, 101.325])
            )
            obses = np.r_[obses, obs[1:3]]
        
        return obses
    
    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[float, dict[str, float]]:
        def base_exp(ref, val, w):
            x = (ref - val)/ref
            
            e = 2.71828182
            y = e**(-w*x*x)
            
            return y
        
        def base_dist(ref, val):
            x = -abs(ref - val)

            return x
        
        def base_log(ref, val):
            x = ref - val
            if abs(x) < 0.999:
                r = math.log(0.001/(abs(x) + 0.001))
            else:
                r = -abs(x) + 0.999 + math.log(0.001)
            return r
        
        def base_coupling(
            pos_ref,
            neg_ref,
            pos_init,
            neg_init,
            pos_press,
            neg_press
        ):
            if [pos_ref, neg_ref] == [pos_init, neg_init]:
                # dp = pos_press - pos_ref
                # dn = neg_press - neg_ref
                # dist = math.sqrt(dp**2 + dn**2)
                dist = 0
            else:
                a = 1/(pos_ref - pos_init + 1e-8)
                b = 1/(neg_ref - neg_init + 1e-8)
                num = abs(a*(pos_ref - pos_press) - b*(neg_ref - neg_press))
                den = math.sqrt(a**2 + b**2)
                dist = num/den

            return dist
        
        def base_dist2(
            pos_ref,
            neg_ref,
            pos_press,
            neg_press
        ):
            dp = pos_press - pos_ref
            dn = neg_press - neg_ref
            dist = math.sqrt(dp**2 + dn**2)

            return dist
        
        def base_diff(state, idx_press):
            press = state[idx_press::]

            pos_idx = np.array([2*i for i in range(self.num_obs)])
            neg_idx = np.array([2*i + 1 for i in range(self.num_obs)])

            pos_press = press[pos_idx]
            neg_press = press[neg_idx]

            pos_diff_reward = - np.sum(np.abs(
                pos_press[1::] - pos_press[0:-1]
            ))
            neg_diff_reward = - np.sum(np.abs(
                neg_press[1::] - neg_press[0:-1]
            ))

            return pos_diff_reward, neg_diff_reward
            
        
        idx_press = self.num_ref*self.dim_ref
        pos_ref = state[0] - 101.325
        neg_ref = state[1] - 101.325
        pos_ref_fut = state[idx_press - 2] - 101.325
        neg_ref_fut = state[idx_press - 1] - 101.325
        pos_press = state[idx_press] - 101.325
        neg_press = state[idx_press + 1] - 101.325
        pos_init = state[-2] - 101.325
        neg_init = state[-1] - 101.325

        action = np.clip(action, -1, 1)
        pos_act = action[0]
        neg_act = action[1]

        # pos_ratio = 1
        # neg_ratio = 3
        # pos_reward = base_dist(pos_ref, pos_press)*pos_ratio/(pos_ratio + neg_ratio)
        # neg_reward = base_dist(neg_ref, neg_press)*neg_ratio/(pos_ratio + neg_ratio)
        pos_ratio = 1
        neg_ratio = 3
        pos_reward = base_dist(pos_ref, pos_press)*pos_ratio/(pos_ratio + neg_ratio)
        neg_reward = base_dist(neg_ref, neg_press)*neg_ratio/(pos_ratio + neg_ratio)
        press_reward = pos_reward + neg_reward

        # press_coeff = 1
        # press_reward = - press_coeff*base_dist2(
        #     pos_ref,
        #     neg_ref,
        #     pos_press,
        #     neg_press
        # )       

        pos_act_coeff = 0
        neg_act_coeff = 0
        pos_act_reward = base_dist(pos_act, -1)*pos_act_coeff
        neg_act_reward = base_dist(neg_act, -1)*neg_act_coeff
        act_reward = pos_act_reward + neg_act_reward

        coup_coeff = 0
        coup_reward = - coup_coeff*base_coupling(
            pos_ref,
            neg_ref,
            pos_ref_fut,
            neg_ref_fut,
            pos_press,
            neg_press
        )

        pos_diff_coeff = 0.25
        neg_diff_coeff = 0.25
        pos_diff_reward, neg_diff_reward = base_diff(state, idx_press)
        pos_diff_reward = pos_diff_coeff*pos_diff_reward
        neg_diff_reward = neg_diff_coeff*neg_diff_reward
        diff_reward = pos_diff_reward + neg_diff_reward

        reward = press_reward + act_reward + diff_reward

        info = {
            'pos_reward': pos_reward,
            'neg_reward': neg_reward,
            'pos_act_reward': pos_act_reward,
            'neg_act_reward': neg_act_reward,
            'pos_diff_reward': pos_diff_reward,
            'neg_diff_reward': neg_diff_reward
        }

        return reward, info
    
    def get_info(
        self,
        state_info: Dict[str, np.ndarray],
        reward_info: Dict[str, float]
    ) -> Dict[str, Union[np.ndarray, float]]:
        return dict(
            obs = state_info['Observation'],
            obs_wo_noise = state_info['obs_w/o_noise'],
            ctrl_input = state_info['ctrl'],
            reward = reward_info
        )
    
    def verbose(
        self,
        info
    ):
        print(
            f'[ INFO] Pneumatic Env ==> \n'
            f'\tTime: {info["obs"]["curr_time"]}\n'
            f'\tSen : (\t{info["obs"]["sen_pos"]:3.4f}\t{info["obs"]["sen_neg"]:3.4f})\n'
            f'\tRef : (\t{info["obs"]["ref_pos"]:3.4f}\t{info["obs"]["ref_neg"]:3.4f})\n'
            f'\tCtrl: (\t{info["obs"]["ctrl_pos"]:3.4f}\t{info["obs"]["ctrl_neg"]:3.4f})\n'
            f'\tC/I : (\t{info["ctrl_input"][0]:3.4f}\t{info["ctrl_input"][1]:3.4f}) \n'
            f'\tw/o : {info["obs_wo_noise"]}\n'
            f'\tRWD : Press\t{info["reward"]["pos_reward"]}\t{info["reward"]["neg_reward"]}\n'
            f'\t      Act\t{info["reward"]["pos_act_reward"]}\t{info["reward"]["neg_act_reward"]}\n'
            f'\t      Diff\t{info["reward"]["pos_diff_reward"]}\t{info["reward"]["neg_diff_reward"]}'
        )
        for _ in range(10):
            sys.stdout.write("\033[F")  # Move the cursor up one line
            sys.stdout.write("\033[K")  # Clear the line
    
    def publish_observation(
        self,
        info: Dict[str, Any]
    ):
        msg = Observation()
        msg.time = info['obs']['curr_time']
        msg.sen_pos = info['obs']['sen_pos']
        msg.sen_neg = info['obs']['sen_neg']
        msg.ref_pos = info['obs']['ref_pos']
        msg.ref_neg = info['obs']['ref_neg']
        msg.ctrl_pos = info['obs']['ctrl_pos']
        msg.ctrl_neg = info['obs']['ctrl_neg']
        self.obs_pub.publish(msg)

        msg = Observation()
        msg.time = info['obs_wo_noise'][0]
        msg.sen_pos = info['obs_wo_noise'][1]
        msg.sen_neg = info['obs_wo_noise'][2]
        msg.ref_pos = info['obs']['ref_pos']
        msg.ref_neg = info['obs']['ref_neg']
        msg.ctrl_pos = info['obs']['ctrl_pos']
        msg.ctrl_neg = info['obs']['ctrl_neg']
        self.obs_wo_noise_pub.publish(msg)

