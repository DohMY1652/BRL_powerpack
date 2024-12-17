from typing import Callable
import numpy as np
from scipy.signal import tf2ss

import matplotlib.pyplot as plt

class TransFunc():
    def __init__(
        self,
        num: list,
        den: list,
        freq: float = 50,
    ):
        self.A, self.B, self.C, self.D = tf2ss(num, den)
        self.dt = 1/freq

        self.x = np.zeros(self.A.shape[0])
        self.u = np.zeros(self.B.shape[0])

        self.pre_process_func = lambda x: x
        self.post_process_func = lambda x: x

    def apply(
        self,
        u: float
    ) -> np.ndarray:
        normalized_u = self.pre_process_func(u)

        self.x = self.x + self.dt * (self.A @ self.x + self.B * normalized_u)
        normalized_y = self.C @ self.x + self.D * normalized_u

        y = self.post_process_func(normalized_y[0][0])
        
        return y

    def set_init_input(
        self,
        u_init: np.ndarray
    ) -> None:
        self.u = np.array([self.pre_process_func(u_init)])
    
    def set_init_state(
        self,
        x_init: np.ndarray,
        pre_process_func: Callable[[float], float]
    ) -> None:
        self.x[0] = np.array([pre_process_func(x_init)])

    def set_pre_process(
        self,
        func: Callable[[float], float]
    ) -> None:
        self.pre_process_func = func
    
    def set_post_process(
        self,
        func: Callable[[float], float]
    ) -> None:
        self.post_process_func = func
    


