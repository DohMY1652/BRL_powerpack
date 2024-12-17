import os

from collections import deque
import numpy as np
import pandas as pd
import time
from datetime import datetime
import pickle
import yaml
import threading

from pneu_ref.random_ref import RandomRef
from pneu_ref.step_ref import StepCasesRef, StepRef
from pneu_ref.sine_ref import SineRef
from pneu_ref.traj_ref import TrajRef
from pneu_ref.ctrl_ref import CtrlTraj
from pneu_env.env import PneuEnv
from pneu_env.sim import PneuSim
# from pneu_env.simulator import PneuSim
from pneu_env.real import PneuReal
from pneu_env.pred import PneuPred
from pneu_rl.sac import SAC
from pneu_utils.utils import (
    delete_lines, 
    color, 
    get_pkg_path,
    load_yaml
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ATM = 101.325

viz_kwargs = dict(
    env = dict(
        sim = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            noise_std = 0,
            offset_pos = 0,
            offset_neg = 0,
            scale = False
        ),
        real = dict(
            freq = 50,
            scale = True
        ),
    )
)

def infos_to_datas(infos: deque):
    datas = dict(
        curr_time = deque(),
        sen_pos = deque(),
        sen_neg = deque(),
        ref_pos = deque(),
        ref_neg = deque(),
        ctrl_pos = deque(),
        ctrl_neg = deque(),
        mf_pos_val = deque(),
        mf_neg_val = deque()
    )
    for info in infos:
        for key, value in info.items():
            datas[key].append(value)
    return datas

def save_datas(datas, model_name, obs_mode, ref_mode, save_name=None, kwargs=None):
    if save_name is not None:
        print('[ INFO] Saving data starts ...')
    
    # obs_mode = "Real" if obs_mode == "2" else "simulation" 

    if save_name is not None:
        os.makedirs(f'{get_pkg_path("pneu_env")}/exp/{save_name}')
        df = pd.DataFrame(datas)
        df.to_csv(f'/Users/greenlandshark/MATLAB/main/datas/{save_name}.csv', index=False)
        df.to_csv(f'{get_pkg_path("pneu_env")}/exp/{save_name}/{save_name}.csv', index=False)

    kwargs["model_name"] = model_name
    kwargs["obs_mode"] = obs_mode
    kwargs["ref_mode"] = ref_mode

    if save_name is not None:
        with open(f'{get_pkg_path("pneu_env")}/exp/{save_name}/cfg.yaml', 'w') as f:
            yaml.dump(kwargs, f)
    
    if save_name is not None:
        print('[ INFO] Saving data Done!')

def plot_datas(datas, save_name=None):
    fontname = 'Times New Roman'
    label_font_size = 18
    
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[2:4,0])
    ax3 = fig.add_subplot(gs[4,0])    

    ax1.plot(
        np.array(datas['curr_time']),
        np.array(datas['ref_pos']),
        linewidth=2, color='black', label='REF'
    )
    ax1.plot(
        np.array(datas['curr_time']),
        np.array(datas['sen_pos']),
        linewidth=2, color='red', label='POS'
    )
    ax1.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
    ax1.set_ylabel('Pressure [kPa]', fontname=fontname, fontsize=label_font_size)
    ax1.grid(which='major', color='silver', linewidth=1)
    ax1.grid(which='minor', color='lightgray', linewidth=0.5)
    ax1.minorticks_on()
    ax1.legend(loc='upper right')
    ax1.set(xlim=(0, None), ylim=(None, None))
    
    ax2.plot(
        np.array(datas['curr_time']),
        np.array(datas['ref_neg']),
        linewidth=2, color='black', label='REF'
    )
    ax2.plot(
        np.array(datas['curr_time']),
        np.array(datas['sen_neg']),
        linewidth=2, color='blue', label='NEG'
    )
    ax2.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
    ax2.set_ylabel('Pressure [kPa]', fontname=fontname, fontsize=label_font_size)
    ax2.grid(True)
    ax2.grid(which='major', color='silver', linewidth=1)
    ax2.grid(which='minor', color='lightgray', linewidth=0.5)
    ax2.minorticks_on()
    ax2.legend(loc='upper right')
    ax2.sharex(ax1)
    ax2.set(xlim=(0, None), ylim=(None, None))

    ax3.plot(
        datas['curr_time'],
        datas['ctrl_pos'],
        linewidth=2, color='red', label='POS'
    )
    ax3.plot(
        datas['curr_time'],
        datas['ctrl_neg'],
        linewidth=2, color='blue', label='NEG'

    )
    ax3.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
    ax3.set_ylabel('Control', fontname=fontname, fontsize=label_font_size)
    ax3.legend(loc='upper right')
    ax3.sharex(ax1)
    ax3.set(xlim=(0, None), ylim=(None, None))

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'{get_pkg_path("pneu_env")}/exp/{save_name}/{save_name}.png')
    # plt.show()

if __name__ == '__main__':
    print('[ INFO] Control Mode: Trajectory')
    print(color('[INPUT] Control Data:', 'blue'))
    
    models = sorted(os.listdir(f"{get_pkg_path('pneu_env')}/exp"))
    for i, model in enumerate(models):
        print(color(f'\t{i+1}. {model}', 'yellow'))
    print(color('\t---', 'blue'))
    model_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
    exp_name = models[model_idx]
    delete_lines(len(models) + 3)
    print(f'[ INFO] Control Data: {exp_name}')

    kwargs = load_yaml(f"../exp/{exp_name}")

    df = pd.read_csv(f"{get_pkg_path('pneu_env')}/exp/{exp_name}/{exp_name}.csv")

    ctrl = CtrlTraj(
        traj_time = df["curr_time"].values,
        traj_pos = df["ctrl_pos"].values,
        traj_neg = df["ctrl_neg"].values
    )
    ref = TrajRef(
        traj_time = df["curr_time"].values,
        traj_pos = df["sen_pos"].values,
        traj_neg = df["sen_neg"].values
    )

    print(color('[INPUT] Observation Mode:', 'blue'))
    print(color('\t1. Sim', 'yellow'))
    print(color('\t2. Real', 'yellow'))
    print(color('\t---', 'blue'))
    obs_mode = input(color('\tOBS: ', 'blue')) 
    obs_type = 'Simulation' if obs_mode == "1" else "Real"
    delete_lines(5)
    print(f'[ INFO] Observation Mode: {"Simulation" if obs_mode == "1" else "Real"}')

    print(color('[INPUT] Save data?', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    data_log = input(color('\tLogging: ', 'blue')) 
    delete_lines(5)
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    if data_log == '1':
        save_name = f'{formatted_time}_TrajCtrl_{obs_type}'
    else:
        save_name = None
    print(f'[ INFO] Data logging: {"False" if data_log == "2" else f"{save_name}.csv"}')

    if obs_mode == '1':    
        obs = PneuSim(**viz_kwargs["env"]["sim"])
        # obs.set_discharge_coeff(3, 3)
        # pred = PneuPred(**viz_kwargs["env"]["pred"])
    elif obs_mode == '2':
        obs = PneuReal(**viz_kwargs["env"]["real"])
        # pred = PneuPred(**viz_kwargs["env"]["pred"])

    obs.set_init_press(
        init_pos_press = df["sen_pos"].values[0],
        init_neg_press = df["sen_neg"].values[0]
    )

    print(color('[INPUT] Discharge coeff:', 'blue'))
    coeff_values = sorted(os.listdir(f"{get_pkg_path('pneu_env')}/data/discharge_coeff_result"))
    for i, coeff_value in enumerate(coeff_values):
        print(color(f'\t{i+1}. {coeff_value}', 'yellow'))
    print(color('\t---', 'blue'))
    coeff_idx = input(color('\tMODEL: ', 'blue'))
    coeff_name = coeff_values[int(coeff_idx) - 1] if len(coeff_idx) != 0 else coeff_values[-1]
    delete_lines(len(coeff_values) + 3)
    print(f'[ INFO] Control Data: {coeff_name}')
    
    with open(f"{get_pkg_path('pneu_env')}/data/discharge_coeff_result/{coeff_name}/coeff.yaml", "r") as f:
        coeff_data = yaml.safe_load(f)
    obs.set_discharge_coeff(**coeff_data)

    try:
        curr_time = 0
        if ref.max_time == float('inf'):
            ref.max_time = 100
        
        infos = deque()
        time_flag = 0
        while curr_time < ref.max_time:
            action = ctrl.get_ctrl(curr_time)
            # action = np.array([-1, -1])
            curr_ref = ref.get_goal(curr_time)
            curr_obs, info = obs.observe(action, curr_ref)
            curr_time = curr_obs[0]
            elapsed_time_flag = curr_time - time_flag
            info = info["Observation"]
            mf = obs.get_mass_flowrate()
            info["mf_pos_val"] = mf[3]
            info["mf_neg_val"] = mf[4]
            infos.append(info)


        datas = infos_to_datas(infos)
        save_datas(datas, exp_name, obs_type, exp_name, save_name, viz_kwargs)
        plot_datas(datas, save_name)

    except KeyboardInterrupt:
        print()
        print(color('[ INFO] Keyboard interrupt received.', 'red'))
        datas = infos_to_datas(infos)
        save_datas(datas, exp_name, obs_type, "Traj", exp_name, viz_kwargs)
        plot_datas(datas, save_name)


    

    

        
