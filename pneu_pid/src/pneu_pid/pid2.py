from typing import Dict
from collections import deque
import numpy as np

from pneu_env.env import PneuEnv

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
    ):
        self.curr_state_idx = np.arange(env.num_prev*env.dim_obs, (env.num_prev + 1)*env.dim_obs)
        self.curr_ref_idx = np.arange(
            (env.num_obs + env.num_prev)*env.dim_obs,
            (env.num_obs + env.num_prev + 1)*env.dim_obs
        )
        
        self.Kp_pos = Kp_pos
        self.Ki_pos = Ki_pos
        self.Kd_pos = Kd_pos
        self.Kp_neg = Kp_neg
        self.Ki_neg = Ki_neg
        self.Kd_neg = Kd_neg
        
        self.sum_pos = 0
        self.sum_neg = 0
        
        self.prev_pos = 0
        self.prev_neg = 0
        
        # DOB parameters
        self.Q = dict(
            pos = TransferFcn(
                num = [0, 0.00299550449662702],
                den = [1, -0.997004495503373]
            ),
            neg = TransferFcn(
                num = [0, 0.00498752080731769],
                den = [1, -0.995012479192682]
            )
        )
        self.QxinvG = dict(
            pos = TransferFcn(
                num = [-0.00312674964314412, 0.00936938213262907, -0.00935852225114769, 0.00311588976143911],
                den = [1, -2.99610983817614, 2.99222234572849, -0.996112507584016]
            ),
            neg = TransferFcn(
                num = [1.03015580635846e-05, 0.00129352430800008, -0.00258187545814816, 0.00127804965943111],
                den = [1, -2.67188945867134, 2.34538902038628, -0.673499569095191]
            )
        )
        self.prev_action = np.array([1, 1], dtype=np.float32)
    
    def predict(
        self, 
        state: np.ndarray, 
        evaluate: bool = True
    ) -> np.ndarray:

        curr_state = state[self.curr_state_idx]
        curr_ref = state[self.curr_ref_idx]

        sen_pos = curr_state[0]
        sen_neg = curr_state[1]

        ref_pos = curr_ref[0]
        ref_neg = curr_ref[1]

        u_lpf_pos = self.Q['pos'].output(self.prev_action[0])
        u_lpf_neg = self.Q['neg'].output(self.prev_action[1])
        
        u_lpf_plus_dhat_pos = self.QxinvG['pos'].output(sen_pos)
        u_lpf_plus_dhat_neg = self.QxinvG['neg'].output(sen_neg)
        
        dhat_pos = u_lpf_plus_dhat_pos - u_lpf_pos
        dhat_neg = u_lpf_plus_dhat_neg - u_lpf_neg
        
        err_pos = - ref_pos + sen_pos
        err_neg = ref_neg - sen_neg
        dt = 0.001

        self.sum_pos += err_pos*dt
        self.sum_neg += err_neg*dt
        
        err_der_pos = (err_pos - self.prev_pos)/(dt + 1e-8)
        err_der_neg = (err_neg - self.prev_neg)/(dt + 1e-8)
        self.prev_pos = err_pos
        self.prev_neg = err_neg
        
        action_pos = \
            self.Kp_pos*err_pos \
            + self.Ki_pos*self.sum_pos \
            + self.Kd_pos*err_der_pos \
            - 0.5*dhat_pos

        action_neg = \
            self.Kp_neg*err_neg \
            + self.Ki_neg*self.sum_neg \
            + self.Kd_neg*err_der_neg \
            - 0.5*dhat_neg
        
        action = np.array([action_pos, action_neg], dtype=np.float32)
        self.prev_action = action
        
        return action

    def predict_from_info(
        self,
        info: Dict[str, np.ndarray]
    ):
        # ref = info['ref']
        ref_pos = info['obs']['ref_pos'] 
        ref_neg = info['obs']['ref_neg']
        
        sen_pos = info['obs']['sen_pos']
        sen_neg = info['obs']['sen_neg']
        
        u_lpf_pos = self.Q['pos'].output(self.prev_action[0])
        u_lpf_neg = self.Q['neg'].output(self.prev_action[1])
        
        u_lpf_plus_dhat_pos = self.QxinvG['pos'].output(sen_pos)
        u_lpf_plus_dhat_neg = self.QxinvG['neg'].output(sen_neg)
        
        dhat_pos = u_lpf_plus_dhat_pos - u_lpf_pos
        dhat_neg = u_lpf_plus_dhat_neg - u_lpf_neg
        
        err_pos = - ref_pos + sen_pos
        err_neg = ref_neg - sen_neg
        dt = 0.001

        self.sum_pos += err_pos*dt
        self.sum_neg += err_neg*dt
        
        err_der_pos = (err_pos - self.prev_pos)/(dt + 1e-8)
        err_der_neg = (err_neg - self.prev_neg)/(dt + 1e-8)
        self.prev_pos = err_pos
        self.prev_neg = err_neg
        
        action_pos = \
            self.Kp_pos*err_pos \
            + self.Ki_pos*self.sum_pos \
            + self.Kd_pos*err_der_pos \
            - 0.5*dhat_pos

        action_neg = \
            self.Kp_neg*err_neg \
            + self.Ki_neg*self.sum_neg \
            + self.Kd_neg*err_der_neg \
            - 0.5*dhat_neg
        
        action = np.array([action_pos, action_neg], dtype=np.float32)
        self.prev_action = action
        
        return action

        
