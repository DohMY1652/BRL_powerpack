import socket
import struct
import threading
from typing import Dict

import numpy as np
import time

import json

from pneu_msg.msg import Observation

class PneuReal():
    def __init__(
        self,
        sen_freq: float = 100,
    ):
        self.sen_period = 1/sen_freq
        
        self.curr_time = 0.0
        self.sen_pos = 101.325
        self.sen_neg = 101.325
        self.ref_pos = 101.325
        self.ref_neg = 101.325
        self.ctrl_pos = 1.0
        self.ctrl_neg = 1.0
        
        self.flag_time = time.time()
        self.start_time = time.time()

    def write_ctrl_file(self) -> None:
        data = dict(
            time = time.time() - self.start_time,
            sen_pos = self.sen_pos,
            sen_neg = self.sen_neg,
            ref_pos = self.ref_pos,
            ref_neg = self.ref_neg,
            ctrl_pos = self.ctrl_pos,
            ctrl_neg = self.ctrl_neg
        )
        with open('json/ctrl.json', 'w') as f:
            json.dump(data, f)
        with open('json/ctrl_backup.json', 'w') as f:
            json.dump(data, f)

    def read_obs_file(self) -> None:
        try:
            with open('json/ctrl.json', 'r') as f:
                obs = json.load(f)
                self.curr_time = obs['time']
                self.sen_pos = obs['sen_pos']
                self.sen_neg = obs['sen_neg']
        except:
            with open('json/ctrl_backup.json', 'r') as f:
                obs = json.load(f)
                self.curr_time = obs['time']
                self.sen_pos = obs['sen_pos']
                self.sen_neg = obs['sen_neg']
    
    def wait(self) -> None:
        curr_flag_time = time.time()
        time.sleep(max(self.sen_period - curr_flag_time + self.flag_time, 0))
        self.flag_time = time.time()

    def get_obs(
        self,
        ctrl: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        ctrl = np.clip(ctrl, -1, 1)
        ctrl = 0.5*ctrl + 0.5

        self.ref_pos = goal[0]
        self.ref_neg = goal[1]
        self.ctrl_pos = ctrl[0]
        self.ctrl_neg = ctrl[1]
        
        self.write_ctrl_file()
        self.wait()
        self.read_obs_file()
        
        next_obs = np.array([
            self.curr_time,
            self.sen_pos,
            self.sen_neg
        ])

        Observation_info = dict(
            curr_time = self.curr_time,
            sen_pos = self.sen_pos,
            sen_neg = self.sen_neg,
            ref_pos = self.ref_pos,
            ref_neg = self.ref_neg,
            ctrl_pos = self.ctrl_pos,
            ctrl_neg = self.ctrl_neg
        )

        info = {}
        info['obs_w/o_noise'] = np.array([
            Observation_info['curr_time'],
            Observation_info['sen_pos'],
            Observation_info['sen_neg']
        ])
        info['Observation'] = Observation_info

        return next_obs, info
    
if __name__ == '__main__':
    env = PneuReal() 

    for _ in range(10):
        obs, info = env.get_obs(
            ctrl = np.array([-1, 1], dtype=np.float64),
            goal = np.array([120, 130], dtype=np.float64)
        )
        print(obs)


    


        
