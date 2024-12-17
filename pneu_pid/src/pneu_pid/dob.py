import numpy as np
from typing import Callable

from pneu_pid.tf import TransFunc

class DisturbanceObserver():
    def __init__(
        self,
        invGxQ: TransFunc,
        Q: TransFunc,
    ):
        self.invGxQ = invGxQ
        self.Q = Q

    def get_disturbance(
        self,
        u: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        return self.invGxQ.apply(y) - self.Q.apply(u)
        