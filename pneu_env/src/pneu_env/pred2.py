from ctypes import CDLL, c_double, POINTER

import numpy as np
from collections import deque

from pneu_utils.utils import get_pkg_path

class PneuPred():
    def __init__(
        self,
        freq: float = 50,
        volume1: float = 0.75,
        volume2: float = 0.4,
        init_pos_press: float = 120,
        init_neg_press: float = 80,
        delay: float = 0,
        noise: bool = False,
        noise_std: float = 1,
        offset_pos: float = 0,
        offset_neg: float = 0,
        scale: bool = False
    ):
        env_pkg_path = get_pkg_path('pneu_env')
        self.lib = CDLL(f'{env_pkg_path}/src/pneu_env/lib/pneumatic_simulator_pred.so')        
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

        self.freq = freq

        self.delay = delay
        self.noise = noise
        self.noise_std = noise_std
        self.offset_pos = offset_pos
        self.offset_neg = offset_neg

        obs_buf_len = int(freq*delay + 1)
        self.obs_buf = deque(maxlen=obs_buf_len)
        self.scale = scale
        print(f'[ INFO] Pneumatic Simulator ==> Delay: {delay}')

    def get_obs(
        self, 
        ctrl: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        ctrl = np.clip(ctrl, -1, 1)
        if self.scale:
            ctrl = 0.3*0.5*(ctrl + 1) + 0.7
        else:
            ctrl = 0.5*ctrl + 0.5
        
        pos_coeff = 16.23565473
        pos_ctrl = ctrl[0]
        pos_ctrl = pos_ctrl**pos_coeff

        neg_coeff = 18.61464263
        neg_ctrl = ctrl[1]
        neg_ctrl = neg_ctrl**neg_coeff

        sim_ctrl = [pos_ctrl, neg_ctrl]

        time_step = 1/self.freq
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
    
    def observe(
        self, 
        ctrl: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325])
    ) -> np.ndarray:
        ctrl = np.clip(ctrl, -1, 1)
        if self.scale:
            ctrl = 0.3*0.5*(ctrl + 1) + 0.7
        else:
            ctrl = 0.5*ctrl + 0.5
        
        pos_coeff = 16.23565473
        pos_ctrl = ctrl[0]
        pos_ctrl = pos_ctrl**pos_coeff

        neg_coeff = 18.61464263
        neg_ctrl = ctrl[1]
        neg_ctrl = neg_ctrl**neg_coeff

        sim_ctrl = [pos_ctrl, neg_ctrl]

        time_step = 1/self.freq
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
    ):
        # self.lib.time_reset()
        self.lib.set_init_env(
            init_pos_press,
            init_neg_press
        )
        # print(f'Press initialized: {init_pos_press}, {init_neg_press}')
    
    def set_offset(
        self,
        pos_offset: float,
        neg_offset: float
    ):
        self.offset_pos = pos_offset
        self.offset_neg = neg_offset
    
    def set_volume(self, vol1, vol2):
        self.lib.set_volume(vol1, vol2)
    
if __name__ == '__main__':
    env = PneuSim()
    ctrl = np.array([0.0, 0.0], dtype=np.float64)
    for n in range(100):
        obs, _ = env.get_obs(ctrl)

        print(obs)

        
