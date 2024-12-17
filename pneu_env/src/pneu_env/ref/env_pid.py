import sys
import rospy

import numpy as np
from typing import Tuple, Dict, Any, Union

from gymnasium.spaces import Box

from pneu_ref.pneu_ref import PneuRef

from pneu_env.pid import PID

class PneuEnv():
    def __init__(
        self,
        obs: Any,
        ref: Any,
        num_prev: int = 10,
        num_pred: int = 10,
        rwd_kwargs: Dict[str, float] = dict(
            pos_prev_rwd_coeff = 0.0,
            neg_prev_rwd_coeff = 0.0,
            pos_curr_rwd_coeff = 0.0,
            neg_curr_rwd_coeff = 0.0,
            pos_fut_rwd_coeff = 0.0,
            neg_fut_rwd_coeff = 0.0,
            pos_pred_rwd_coeff = 0.0,
            neg_pred_rwd_coeff = 0.0,
            pos_diff_rwd_coeff = 0.0,
            neg_diff_rwd_coeff = 0.0,
        )
    ):
        self.obs = obs
        self.goal = PneuRef(
            ref,
            num_prev = num_prev,
            num_pred = num_pred,
            ctrl_freq = self.obs.freq
        )
        
        # Definition of Env dimensions
        self.num_prev = num_prev
        self.num_pred = num_pred
        self.num_act = 1
        
        self.num_obs = num_prev + 1
        self.num_ref = num_prev + num_pred + 1

        self.dim_obs = 2
        self.dim_act = 2

        self.dim_obs_traj = self.num_obs*self.dim_obs
        self.dim_ref_traj = self.num_ref*self.dim_obs
        self.dim_act = self.num_act*self.dim_act
        self.dim_state = self.dim_obs_traj + self.dim_ref_traj
        
        # Env dimension
        self.observation_space = Box(
            low = -np.inf,
            high = np.inf,
            shape = (self.dim_state, ),
            dtype = np.float64
        )
        self.action_space = Box(
            low = np.array([101.325, 0.0]),
            high = np.array([500, 101.325]),
            shape = (self.dim_act, ),
            dtype = np.float64
        )

        # Initialize parameters
        self.obs_traj = 101.325*np.ones((self.num_obs, self.dim_obs), dtype=np.float64)
        self.t = 0.0
        self.rwd_kwargs = rwd_kwargs
        self.is_pid = False
        self.curr_obs = 101.325*np.ones((self.dim_obs), dtype=np.float32)
        self.curr_ref = 101.325*np.ones((self.dim_obs), dtype=np.float32)

    def set_pid(
        self,
        Kp_pos: float = 0.1,
        Ki_pos: float = 0,
        Kd_pos: float = 0.05,
        Kp_neg: float = 0.1,
        Ki_neg: float = 0.0,
        Kd_neg: float = 0.05,
    ) -> None:
        self.is_pid = True
        self.pid = PID(
            Kp_pos = Kp_pos,
            Ki_pos = Ki_pos,
            Kd_pos = Kd_pos,
            Kp_neg = Kp_neg,
            Ki_neg = Ki_neg,
            Kd_neg = Kd_neg,
            freq = self.obs.freq
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctrl = self.curr_ref*np.ones(self.action_space.shape[0], dtype=np.float64)
        state, _, _, _, info = self.step(ctrl)
        self.t = info['obs']['curr_time']

        if self.is_pid:
            self.pid.reset()
        
        return state, info 
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        if self.is_pid:
            action_pid = self.pid.get_action(self.curr_obs, action)
        state, state_info = self.get_state(action_pid, action)
        reward, reward_info = self.get_reward(
            state = state,
            action = action_pid
        )
        terminated = False
        truncated = False

        info = self.get_info(state_info, reward_info) 
        
        self.verbose(info)

        return (
            state,
            reward,
            terminated,
            truncated,
            info
        )
        
    def close(self):
        ctrl = np.ones(self.dim_act, dtype=np.float64)
        goal = 101.325*np.ones(self.dim_obs, dtype=np.float64)
        _, _ = self.obs.get_obs(ctrl, goal)

    def get_state(
        self,
        ctrl: np.ndarray,
        action: np.ndarray # Calculated reference
    ) -> np.ndarray:
        ctrl_traj = ctrl.reshape(-1, self.dim_act)
    
        # Reshape [[pos, neg], [pos, neg], ... ]
        goal_traj = self.goal.get_ref(self.t).reshape(-1,self.dim_obs) # next reference traj of time t
        self.curr_ref = goal_traj[0]

        # Get next observation o_{t+1}
        obs, obs_info = self.obs.get_obs(ctrl_traj.copy()[0], goal_traj.copy()[self.num_obs])
        obs_traj = np.r_[
            self.obs_traj[1:],
            obs[1:3].reshape(-1,self.dim_obs)
        ]
        self.curr_obs = obs[1:3]

        state = np.r_[
            obs_traj.reshape(-1), 
            goal_traj.reshape(-1)
        ]

        # Update parameters
        self.t = obs[0] # t -> t+1
        self.obs_traj = obs_traj
        obs_info['ctrl'] = ctrl
        obs_info['action'] = action

        return state, obs_info
    
    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, dict[str, float]]:
        
        state = state.reshape(-1, self.dim_obs)
        
        obses = state[0:self.num_obs]
        refs = state[self.num_obs + self.num_act:2*self.num_obs + self.num_act]

        errs = refs - obses
        prev_errs = errs[0:-1]
        curr_err = errs[-1]
        act_err = refs[0] - action

        reward = 0

        pos_prev_reward = np.sum(np.abs(prev_errs[:,0]))
        pos_prev_reward *= - self.rwd_kwargs['pos_prev_rwd_coeff']
        neg_prev_reward = np.sum(np.abs(prev_errs[:,1]))
        neg_prev_reward *= - self.rwd_kwargs['neg_prev_rwd_coeff']
        reward += pos_prev_reward + neg_prev_reward

        pos_curr_reward = np.abs(curr_err[0])
        pos_curr_reward *= - self.rwd_kwargs['pos_curr_rwd_coeff']
        neg_curr_reward = np.abs(curr_err[1])
        neg_curr_reward *= - self.rwd_kwargs['neg_curr_rwd_coeff']
        reward += pos_curr_reward + neg_curr_reward

        pos_act_reward = np.abs(act_err[0])
        pos_act_reward *= - self.rwd_kwargs['pos_pred_rwd_coeff']
        neg_act_reward = np.abs(act_err[1])
        neg_act_reward *= - self.rwd_kwargs['neg_pred_rwd_coeff']
        reward += pos_act_reward + neg_act_reward
        
        info = {
            'pos_prev_reward': pos_prev_reward,
            'neg_prev_reward': neg_prev_reward,
            'pos_curr_reward': pos_curr_reward,
            'neg_curr_reward': neg_curr_reward,
            'pos_fut_reward': 0,
            'neg_fut_reward': 0,
            'pos_pred_reward': pos_act_reward,
            'neg_pred_reward': neg_act_reward,
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
            action_input = state_info['action'],
            pred = None,
            reward = reward_info
        )
    
    def verbose(
        self,
        info
    ):
        # np.set_printoptions(precision=4,linewidth=80, suppress=True)
        print(
            f'[ INFO] Pneumatic Env ==> \n'
            f'\tTime: {info["obs"]["curr_time"]}\n'
            f'\tSen : (\t{info["obs"]["sen_pos"]:3.4f}\t{info["obs"]["sen_neg"]:3.4f})\n'
            f'\tRef : (\t{info["obs"]["ref_pos"]:3.4f}\t{info["obs"]["ref_neg"]:3.4f})\n'
            f'\tPID : (\t{info["action_input"][0]:3.4f}\t{info["action_input"][1]:3.4f})\n'
            f'\tCtrl: (\t{info["obs"]["ctrl_pos"]:3.4f}\t{info["obs"]["ctrl_neg"]:3.4f})\n'
            f'\tC/I : (\t{info["ctrl_input"][0]:3.4f}\t{info["ctrl_input"][1]:3.4f}) \n'
            f'\tw/o : {info["obs_wo_noise"]}\n'
            f'\tRWD : Curr \t{info["reward"]["pos_curr_reward"]:.4f}\t{info["reward"]["neg_curr_reward"]:.4f}\n'
            f'\t    : Prev \t{info["reward"]["pos_prev_reward"]:.4f}\t{info["reward"]["neg_prev_reward"]:.4f}\n'
            f'\t    : Fut  \t{info["reward"]["pos_fut_reward"]:.4f}\t{info["reward"]["neg_fut_reward"]:.4f}\n'
            f'\t    : Pred \t{info["reward"]["pos_pred_reward"]:.4f}\t{info["reward"]["neg_pred_reward"]:.4f}\n'
        )
            # f'\tPRED:\n'
            # f'   ACT              PRESS             REF\n'
            # f'{np.hstack((info["pred"]["pred_act"].reshape(-1,2), info["pred"]["pred_press"].reshape(-1,2), info["pred"]["pred_ref"].reshape(-1,2)))}'
            # f'\t      Act\t{info["reward"]["pos_act_reward"]}\t{info["reward"]["neg_act_reward"]}\n'
            # f'\t      Diff\t{info["reward"]["pos_diff_reward"]}\t{info["reward"]["neg_diff_reward"]}'
        # )
        for _ in range(13):
            sys.stdout.write('\x1b[1A')  # move the cursor up one line
            sys.stdout.write('\x1b[2K')  # clear the line
        # for _ in range(8):
        #     sys.stdout.write("\033[f")  # move the cursor up one line
        #     sys.stdout.write("\033[k")  # clear the line
    
    def set_volume(self, vol1, vol2):
        self.obs.set_volume(vol1, vol2)


if __name__ == '__main__':
    from pneu_ref.random_ref import RandomRef
    from sim import PneuSim
    from pred import PneuPred
    obs = PneuSim()
    ref = RandomRef()
    pred = PneuPred()
    env = PneuEnv(
        obs = obs,
        ref = ref,
        pred = pred
    )

    action = np.ones(env.action_space.shape, dtype=np.float64)
    state = env.reset()[0]
    print(env.step(action))

