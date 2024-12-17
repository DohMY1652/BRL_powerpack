from typing import Dict, List, Callable
from collections import deque
import numpy as np

from pneu_pid.dob import DisturbanceObserver
from pneu_pid.tf import TransFunc
from pneu_env.env import PneuEnv
from pneu_utils.utils import color

class PID():
    def __init__(
        self,
        env: PneuEnv,
        Kp_pos: float = 1,
        Ki_pos: float = 1,
        Kd_pos: float = 0,
        Kp_neg: float = 1,
        Ki_neg: float = 1,
        Kd_neg: float = 0,
        freq: float = 50
    ):
        self.curr_state_idx = np.arange(env.num_prev*env.dim_obs, (env.num_prev + 1)*env.dim_obs)
        self.curr_ref_idx = np.arange(
            (env.num_obs + env.num_prev)*env.dim_obs,
            (env.num_obs + env.num_prev + 1)*env.dim_obs
        )
        
        self.Kp = np.array([Kp_pos, Kp_neg], dtype=np.float32)
        self.Ki = np.array([Ki_pos, Ki_neg], dtype=np.float32)
        self.Kd = np.array([Kd_pos, Kd_neg], dtype=np.float32)
        
        self.sum = np.array([0, 0], dtype=np.float32)
        self.prev = np.array([0, 0], dtype=np.float32)

        self.dt = 1/freq

        self.is_anti_windup = False
        self.is_dob = False
        self.is_ff = False
        self.scale = False
    
    def predict(
        self, 
        state: np.ndarray,
        evaluate: bool = True
    ) -> np.ndarray:
        curr_obs = state[self.curr_state_idx]
        curr_ref = state[self.curr_ref_idx]

        action = self.get_action(curr_obs, curr_ref)

        return action

    def get_action(
        self, 
        obs: np.ndarray, 
        ref: np.ndarray
    ) -> np.ndarray:
        self.ff_u = self.feedforward(ref) if self.is_ff else np.array([0, 0])
    
        err = ref - obs
        err = err*np.array([-1, 1], dtype=np.float32)
        self.sum += err*self.dt
        err_der = (err - self.prev)/(self.dt + 1e-8)
        
        dist = np.array([0., 0.])

        if self.is_dob:
            self.dist_pos = self.dob_pos.get_disturbance(self.action[0], obs[0])
            self.dist_neg = self.dob_neg.get_disturbance(self.action[1], obs[1])
            self.dist_tot = self.Kdob*np.array([self.dist_pos, self.dist_neg])
            self.dist_com = self.Kcom*np.array([self.dist_neg, self.dist_pos])
            dist = self.dist_tot - self.dist_com
        
        action = \
            self.ff_u \
            + self.Kp*err \
            + self.Ki*self.sum \
            + self.Kd*err_der \
            - dist
        
        self.action = action
        self.prev = err

        return action
    
    def anti_windup(
        self,
        ctrl: np.ndarray,
        sat_ctrl: np.ndarray
    ) -> None:
        ctrl_err = ctrl - sat_ctrl
        self.sum -= self.Ka*ctrl_err
        
    def feedforward(
        self,
        ref: np.ndarray
    ) -> np.ndarray:
        ref_pos = ref[0]
        ref_neg = ref[1]
        
        pos_ff = self.pos_ff_func(ref_pos)
        neg_ff = self.neg_ff_func(ref_neg)
        
        u = 2*np.array([pos_ff, neg_ff]) - 1

        return u

    def reset(self) -> None:
        self.sum = np.array([0, 0], dtype=np.float32)
        self.prev = np.array([0, 0], dtype=np.float32)
    
    def set_anti_windup(
        self,
        Ka: float,
    ) -> None:
        self.is_anti_windup = True
        self.Ka = Ka
    
    def set_disturbance_observer(
        self,
        dob_pos: DisturbanceObserver,
        dob_neg: DisturbanceObserver,
        Kdob: np.ndarray = np.array([1., 1.], np.float32),
        Kcom: np.ndarray = np.array([1., 1.], np.float32)
    ) -> None:
        self.is_dob = True
        self.dob_pos = dob_pos
        self.dob_neg = dob_neg
        self.Kdob = Kdob
        self.Kcom = Kcom
        self.action = np.array([1., 1.])
    
    def set_feedforward(
        self,
        pos_type: str = "exp2",
        pos_coeff: List[float] = [3692.5085,-0.096139,0.98873,-0.0004336],
        neg_type: str = "poly3",
        neg_coeff: List[float] = [1.2799e-06,-0.00017835,0.0087461,0.79111]
    ):
        self.is_ff = True
        self.pos_ff_func = self.get_feedforward_func(pos_type, pos_coeff)
        self.neg_ff_func = self.get_feedforward_func(neg_type, neg_coeff)
    
    def get_feedforward_func(
        self,
        ff_type: str,
        coeff: List[float]
    ) -> Callable[[float], float]:
        if ff_type == "exp2":
            assert len(coeff) == 4, color("[ERROR] Length of exp2's coefficient should be 4.")
            func = lambda x: coeff[0]*np.exp(coeff[1]*x) + coeff[2]*np.exp(coeff[3]*x)
        
        if ff_type == "poly3":
            assert len(coeff) == 4, color("[ERROR] Length of exp2's coefficient should be 4.")
            func = lambda x: coeff[0]*x**3 + coeff[1]*x**2 + coeff[2]*x + coeff[3]
        
        return func
    
    def get_disturbance(self):
        return self.dist_pos, self.dist_neg, self.dist_tot, self.dist_com
    
    def get_feedforward(self):
        return self.ff_u
        
    
if __name__ == '__main__':
    pid = PID()
    
    action = pid.predict(
        np.array([1, 1]),
        np.array([0.99, 0.99])
    )
    pid.reset()
    print(pid.sum)