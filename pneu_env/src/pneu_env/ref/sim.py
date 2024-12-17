import sys
import rospy

from ctypes import CDLL, c_double, POINTER

import numpy as np
import math
from typing import Tuple
from collections import deque

from gymnasium.spaces import Box
from pneu_utils.utils import get_pkg_path

class PneuSim():
    def __init__(
        self,
        sen_freq: float = 100,
        volume1: float = 0.75,
        volume2: float = 0.4,
        # init_pos_press: float = 138,
        # init_neg_press: float = 30,
        init_pos_press: float = 139.2,
        init_neg_press: float = 41.3,
        delay: float = 0,
        noise: bool = False,
        noise_std: float = 1,
        offset_pos: float = 0,
        offset_neg: float = 0
    ):
        env_pkg_path = get_pkg_path('pneu_env')
        self.lib = CDLL(f'{env_pkg_path}/src/pneu_env/lib/pneumatic_simulator.so')        
        self.lib.set_init_env.argtypes = [c_double, c_double]
        self.lib.set_volume.argtypes = [c_double, c_double]
        self.lib.get_time.restype = c_double
        self.lib.step.argtypes = [POINTER(c_double), c_double]
        self.lib.step.restype = POINTER(c_double)
        self.lib.set_discharge_coeff.argtypes = [c_double for _ in range(8)]

        self.lib.set_volume(volume1, volume2)
        
        self.init_pos_press = init_pos_press
        self.init_neg_press = init_neg_press
        self.lib.set_init_env(
            init_pos_press,
            init_neg_press
        )

        self.sen_freq = sen_freq

        self.set_sim_option(
            delay = delay,
            noise = noise,
            noise_std = noise_std,
            offset_pos = offset_pos,
            offset_neg = offset_neg
        )

        obs_buf_len = int(sen_freq*delay + 1)
        self.obs_buf = deque(maxlen=obs_buf_len)
        print(f'[ INFO] Pneumatic Simulator ==> Delay: {delay}')

    def get_obs(
        self, 
        ctrl: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        ctrl = np.clip(ctrl, -1, 1)

        pos_coeff = 21.40166056
        pos_ctrl = 0.5*ctrl[0] + 0.5
        pos_ctrl = (math.exp(pos_coeff*pos_ctrl) - 1)/(math.exp(pos_coeff) - 1)

        neg_coeff = 9.90056976
        neg_ctrl = 0.5*ctrl[1] + 0.5
        neg_ctrl = (math.exp(neg_coeff*neg_ctrl) - 1)/(math.exp(neg_coeff) - 1)

        sim_ctrl = [pos_ctrl, neg_ctrl]

        time_step = 1/self.sen_freq
        next_obs = np.array(
            list(self.lib.step((c_double*2)(*sim_ctrl), time_step)[0:3]),
            dtype=np.float64
        )

        self.obs_buf.append(next_obs)
        next_obs = self.obs_buf[0]

        info = {}
        info['obs_w/o_noise'] = next_obs.copy() 

        if self.noise:
            noise = np.random.normal(0, self.noise_std, 2)
            noise = np.concatenate(([0], noise), axis=0)
            next_obs += noise
            next_obs += np.array([0, self.offset_pos, self.offset_neg])

        Observation_info = dict(
            curr_time = next_obs[0],
            sen_pos = next_obs[1],
            sen_neg = next_obs[2],
            ref_pos = goal[0],
            ref_neg = goal[1],
            ctrl_pos = 0.5*ctrl[0] + 0.5,
            ctrl_neg = 0.5*ctrl[1] + 0.5
        )
        info['Observation'] = Observation_info

        return next_obs, info 
    
    def set_init_press(
        self,
        init_pos_press: float,
        init_neg_press: float
    ) -> None:
        self.lib.set_init_env(
            init_pos_press,
            init_neg_press
        )
    
    def set_sim_option(
        self,
        delay: float = 0,
        noise: bool = False,
        noise_std: float = 1,
        offset_pos: float = 0,
        offset_neg: float = 0
    ) -> None:
        self.delay = delay
        self.noise = noise
        self.noise_std = noise_std
        self.offset_pos = offset_pos
        self.offset_neg = offset_neg

    
if __name__ == '__main__':
    env = PneuSim()
    ctrl = np.array([0.0, 0.0], dtype=np.float64)
    for n in range(100):
        obs, _ = env.get_obs(ctrl)

        print(obs)

        
