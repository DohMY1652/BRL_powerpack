import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from typing import Dict, List

from pneu_env.sim import PneuSim
from pneu_env.pred import PneuPred
from pneu_ref.ctrl_ref import CtrlTraj

def get_ctrl_data(data_file: str):
    data = pd.read_csv(data_file)
    traj_time = data['curr_time'].to_numpy()
    traj_pos = data['ctrl_pos'].to_numpy()
    traj_neg = data['ctrl_neg'].to_numpy()

    return traj_time, traj_pos, traj_neg
    
def collect_data(
    sim: PneuSim, 
    pred: PneuPred, 
    ctrl: CtrlTraj, 
    pred_step: int
):
    pred_data = []
    
    sim_curr_time_lst = []
    sim_sen_pos_lst = []
    sim_sen_neg_lst = []

    curr_time = 0
    curr_pos = 120
    curr_neg = 80
    while curr_time < ctrl.max_time:
        pred_curr_time_lst = []
        pred_sen_pos_lst = []
        pred_sen_neg_lst = []

        pred.set_init_press(curr_pos, curr_neg)
        for _ in range(pred_step):
            if curr_time > ctrl.max_time:
                break

            next_obs, _ = sim.get_obs(ctrl.get_ctrl(curr_time), np.array([101.325, 101.325]))
            pred_obs, _ = pred.get_obs(ctrl.get_ctrl(curr_time), np.array([101.325, 101.325]))

            curr_time = next_obs[0]
            curr_pos = next_obs[1]
            curr_neg = next_obs[2]

            sim_curr_time_lst.append(next_obs[0].item())
            sim_sen_pos_lst.append(next_obs[1].item())
            sim_sen_neg_lst.append(next_obs[2].item())

            pred_curr_time_lst.append(pred_obs[0].item())
            pred_sen_pos_lst.append(pred_obs[1].item())
            pred_sen_neg_lst.append(pred_obs[2].item())
        
        pred_step_data = dict(
            curr_time = np.array(pred_curr_time_lst),
            sen_pos = np.array(pred_sen_pos_lst),
            sen_neg = np.array(pred_sen_neg_lst)
        )
        pred_data.append(pred_step_data)
    
    sen_data = dict(
        curr_time = np.array(sim_curr_time_lst),
        sen_pos = np.array(sim_sen_pos_lst),
        sen_neg = np.array(sim_sen_neg_lst)
    )
    
    return sen_data, pred_data

def plot_data(
    sim_data: Dict[str, np.ndarray],
    pred_data: List[Dict[str, np.ndarray]]
):
    plt.figure()
    plt.plot(sim_data['curr_time'], sim_data['sen_pos'])
    plt.plot(sim_data['curr_time'], sim_data['sen_neg'])
    for d in pred_data:
        plt.plot(d['curr_time'], d['sen_pos'])
        plt.plot(d['curr_time'], d['sen_neg'])
    plt.show()
        

if __name__ == '__main__':
    sim = PneuSim(freq=50, delay=0.1, scale=True)
    pred = PneuPred(freq=50, delay=0.1, scale=True)

    data_file = '240603_14_06_17_C_v00_Simulation.csv'
    traj_time, traj_pos, traj_neg = get_ctrl_data(data_file)
    ctrl = CtrlTraj(traj_time, traj_pos, traj_neg)
    
    sim_data, pred_data = collect_data(sim, pred, ctrl, 10)
    plot_data(sim_data, pred_data)