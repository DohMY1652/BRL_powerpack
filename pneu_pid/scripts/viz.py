import os

from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

from pneu_ref.random_ref import RandomRef
from pneu_ref.step_ref import StepCasesRef, StepRef
from pneu_ref.sine_ref import SineRef, DynamicOscillatorRef, CenterStepOscillationRef
from pneu_ref.traj_ref import TrajRef
from pneu_env.env import PneuEnv
from pneu_env.sim import PneuSim
from pneu_env.real import PneuReal
from pneu_pid.pid import PID
from pneu_pid.utils import get_dobs
from pneu_utils.utils import (
    delete_lines, 
    color, 
    get_pkg_path,
    load_yaml
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ATM = 101.325

Ku_pos = 0.015
Tu_pos = 2
Ku_neg = 0.011
Tu_neg = 1

kwargs = dict(
    freq = 50,
    gains = dict(
        # Kp_pos = Ku_pos,
        # Ki_pos = 0,
        # Kd_pos = 0,
        # Kp_neg = Ku_neg,
        # Ki_neg = 0,
        # Kd_neg = 0,

        # ==> Classic PID <==
        Kp_pos = 0.6*Ku_pos,
        Ki_pos = 1.2*Ku_pos/Tu_pos,
        Kd_pos = 0.075*Ku_pos*Tu_pos,
        Kp_neg = 0.6*Ku_neg,
        Ki_neg = 1.2*Ku_neg/Tu_neg,
        Kd_neg = 0.075*Ku_neg*Tu_neg,

        # # ==> PI <==
        # Kp_pos = 0.45*Ku_pos,
        # Ki_pos = 0.54*Ku_pos/Tu_pos,
        # Kd_pos = 0.0,
        # Kp_neg = 0.45*Ku_neg,
        # Ki_neg = 0.54*Ku_neg/Tu_neg,
        # Kd_neg = 0.0,

        # # ==> Some overshoot <==
        # Kp_pos = Ku_pos/3,
        # Ki_pos = 2/3*Ku_pos/Tu_pos,
        # Kd_pos = 1/9*Ku_pos*Tu_pos,
        # Kp_neg = Ku_neg/3,
        # Ki_neg = 2/3*Ku_neg/Tu_neg,
        # Kd_neg = 1/9*Ku_neg*Tu_neg,

        # # ==> No overshoot <==
        # Kp_pos = 0.2*Ku_pos,
        # Ki_pos = 0.4*Ku_pos/Tu_pos,
        # Kd_pos = 0.06666666*Ku_pos*Tu_pos,
        # Kp_neg = 0.2*Ku_neg,
        # Ki_neg = 0.4*Ku_neg/Tu_neg,
        # Kd_neg = 0.0666666*Ku_neg*Tu_neg,
    ),
    anti_windup = dict(
        Ka = 10
    ),
    dob = dict(
        Kdob_pos = 0.0001,
        Kdob_neg = 0.0001
    ),
    ff = dict(
        pos_type = 'exp2',
        pos_coeff = [23696973680.4878,-0.19559,1.0598,-0.00052646],
        neg_type = 'exp2',
        neg_coeff = [0.89784,0.0028628,-184075.4861,-1.3731]
        # pos_type = 'exp2',
        # pos_coeff = [0, 0, 0, 0],
        # neg_type = 'poly3',
        # neg_coeff = [0, 0, 0, 0]
    ),
    ref = dict(
        sine = dict(
            pos_amp = 10,
            pos_per = 6,
            pos_off = 60 + ATM,
            neg_amp = 5,
            neg_per = 4,
            neg_off = - 45 + ATM,
            iter = 2
        ),
        traj = dict(
            # file = "Pos_Neg_MPC_w_SH_v12_24_07_02"
            # file = "240709_16_35_05_U_v05_Real"
            file = "241114_10_50_47_Ours000_v09_Real"
        ),
        dynamic = dict(
            # trans_time = 40,
            # pos_init_press = 50 + ATM,
            # pos_final_press = 120 + ATM,
            # pos_amp = 10,
            # pos_per = 5,
            # neg_init_press = - 40 + ATM,
            # neg_final_press = - 70 + ATM,
            # neg_amp = 5, 
            # neg_per = 6,
            # trans_time = 100,
            # pos_init_press = 80 + ATM,
            # pos_final_press = 110 + ATM,
            # pos_amp = 5,
            # pos_per = 6,
            # neg_init_press = - 70 + ATM,
            # neg_final_press = - 80 + ATM,
            # neg_amp = 3, 
            # neg_per = 7,
            trans_time = 60,
            pos_init_press = 80 + ATM,
            pos_final_press = 110 + ATM,
            pos_amp = 5,
            pos_per = 8,
            neg_init_press = - 70 + ATM,
            neg_final_press = - 80 + ATM,
            neg_amp = 3, 
            neg_per = 10,
        ),
        center_step = dict(
            trans_time = 40,
            pos_time_step = 10,
            pos_center_step = 15,
            pos_amp = 5,
            pos_per = 5,
            pos_init_press = 50 + ATM,
            neg_time_step = 10,
            neg_center_step = - 5,
            neg_amp = 2,
            neg_per = 6,
            neg_init_press = - 45 + ATM,
        )
    )
)

print()
print(color('[ INFO] PID Gains', 'blue'))
print(color(f'\tP > POS: {kwargs["gains"]["Kp_pos"]:.4f} NEG: {kwargs["gains"]["Kp_neg"]:.4f}', 'yellow'))
print(color(f'\tI > POS: {kwargs["gains"]["Ki_pos"]:.4f} NEG: {kwargs["gains"]["Ki_neg"]:.4f}', 'yellow'))
print(color(f'\tD > POS: {kwargs["gains"]["Kd_pos"]:.4f} NEG: {kwargs["gains"]["Kd_neg"]:.4f}', 'yellow'))
print(color('\t---', 'blue'))
print()

def infos_to_datas(infos: deque):
    datas = dict(
        curr_time = deque(),
        sen_pos = deque(),
        sen_neg = deque(),
        ref_pos = deque(),
        ref_neg = deque(),
        ctrl_pos = deque(),
        ctrl_neg = deque()
    )
    for info in infos:
        for key, value in info.items():
            datas[key].append(value)
    return datas

def save_datas(datas, model_name, obs_mode, ref_mode, kwargs, save_name=None):
    if save_name is not None:
        print('[ INFO] Saving data starts ...')
    
    obs_mode = "Simulation" if obs_mode == "1" else "Real" 

    if save_name is not None:
        os.makedirs(f'{get_pkg_path("pneu_pid")}/exp/{save_name}')
        df = pd.DataFrame(datas)
        df.to_csv(f'/Users/greenlandshark/MATLAB/main/datas/{save_name}.csv', index=False)
        df.to_csv(f'{get_pkg_path("pneu_pid")}/exp/{save_name}/{save_name}.csv', index=False)

    kwargs["model_name"] = model_name
    kwargs["obs_mode"] = obs_mode
    kwargs["ref_mode"] = ref_mode

    if save_name is not None:
        with open(f'{get_pkg_path("pneu_pid")}/exp/{save_name}/cfg.yaml', 'w') as f:
            yaml.dump(kwargs, f)
    
    if save_name is not None:
        print('[ INFO] Saving data Done!')

def plot_datas(datas, save_name=None):
    fontname = 'Times New Roman'
    label_font_size = 18
    fig_name = 'fig'
    
    # fig = plt.figure(figsize=(5, 5))
    fig = plt.figure(figsize=(10, 7))
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
        plt.savefig(f'{get_pkg_path("pneu_pid")}/exp/{save_name}/{save_name}.png')
    plt.show()

if __name__ == '__main__':
    print('[ INFO] Control Mode: PID')
    model_name = "PID"
    
    print(color('[INPUT] Reference Mode:', 'blue'))
    print(color('\t1. Step case', 'yellow'))
    print(color('\t2. Random', 'yellow'))
    print(color('\t3. Trajectory', 'yellow'))
    print(color('\t4. Sinusoidal', 'yellow'))
    print(color('\t5. Paper(dynamic oscillator)', 'yellow'))
    print(color('\t---', 'blue'))
    ref_mode = input(color('\tREF: ', 'blue')) 
    # ref_mode = '1'
    delete_lines(8)
    
    if ref_mode == '1':
        print(f'[ INFO] Reference Mode: Step case')
        ref = StepCasesRef(
            time_step = 5,
            ref_pos_max = 180,
            ref_pos_min = 160,
            ref_neg_max = 70,
            ref_neg_min = 50
        )
    elif ref_mode == '2': 
        print(f'[ INFO] Reference Mode: Random')
        ref = RandomRef(
            # **kwargs['rnd_ref']
            pos_max_off = 200,
            pos_min_off = 150,
            neg_max_off = 70,
            neg_min_off = 40,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 10,
            neg_max_amp = 10,
            seed = 61098

        )
    elif ref_mode =='3':
        print(f'[ INFO] Reference Mode: Trajectory')
        csv_file_name = kwargs["ref"]["traj"]["file"]
        csv_data = pd.read_csv(f"{csv_file_name}.csv").to_dict(orient="list")
        keys = [
            "curr_time",
            "sen_pos",
            "sen_neg",
            "ref_pos",
            "ref_neg",
        ]
        dict_data = {}
        for k, v in zip(keys, csv_data.values()):
            dict_data[k] = np.array(v)
        dict_data["curr_time"] -= dict_data["curr_time"][0]
        # dict_data["ref_pos"] += ATM
        # dict_data["ref_neg"] += ATM

        ref = TrajRef(
            traj_time = dict_data["curr_time"],
            traj_pos = dict_data["ref_pos"],
            traj_neg = dict_data["ref_neg"]
        )
        ref_type = "trajectory"
    elif ref_mode == '4':
        print(f'[ INFO] Reference Mode: Sinusoidal')
        ref = SineRef(**kwargs["ref"]["sine"])

    elif ref_mode == '5':
        print(f'[ INFO] Reference Mode: dynamic oscillator')
        ref = DynamicOscillatorRef(**kwargs['ref']['dynamic'])
        ref_type = "dynamic_oscillator"
        # ref = CenterStepOscillationRef(**kwargs['ref']['center_step'])
        # ref_type = "center_step"
    
    print(color('[INPUT] Observation Mode:', 'blue'))
    print(color('\t1. Sim', 'yellow'))
    print(color('\t2. Real', 'yellow'))
    print(color('\t---', 'blue'))
    obs_mode = input(color('\tOBS: ', 'blue')) 
    # obs_mode = '2'
    delete_lines(5)
    print(f'[ INFO] Observation Mode: {"Simulation" if obs_mode == "1" else "Real"}')

    print(color('[INPUT] Save data?', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    data_log = input(color('\tLogging: ', 'blue')) 
    # data_log = '2'
    delete_lines(5)
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    if data_log == '1':
        obs_type = 'Simulation' if obs_mode == "1" else "Real"
        save_name = f'{formatted_time}_{model_name}_{obs_type}'
    else:
        save_name = None
    print(f'[ INFO] Data logging: {"False" if data_log == "2" else f"{save_name}.csv"}')

    if obs_mode == '1':    
        obs = PneuSim(
            freq = 50,
            delay = 0.1,
            noise = False,
            noise_std = 0,
            offset_pos = 0,
            offset_neg = 0,
            scale = False
            # scale = False
        )
    elif obs_mode == '2':
        obs = PneuReal(
            freq = 50,
            scale = False
        )
        
    obs.set_init_press(
        init_pos_press = 170,
        init_neg_press = 50
    )
    
    env = PneuEnv(
        obs = obs,
        ref = ref,
        num_prev = 1,
        num_act = 1,
        num_pred = 1
    )
    
    print()
    print("[ INFO] PID Setting")
    model = PID(env, **kwargs['gains'])
    print()
    
    print(color('[INPUT] PID ==> Anti-windup:', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    # is_anti_windup = input(color('\tAnti-windup: ', 'blue')) 
    is_anti_windup = '1'
    delete_lines(5)
    if is_anti_windup == "1":
        print(f'[ INFO] PID ==> Anti windup: Yes')
        model.set_anti_windup(**kwargs["anti_windup"])
    else:
        print(f'[ INFO] PID ==> Anti windup: No')
    
    print(color('[INPUT] PID ==> Disturbance Observer:', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    # is_dob = input(color('\tDOB: ', 'blue')) 
    is_dob = '1'
    delete_lines(5)
    if is_dob == "1":
        print(f'[ INFO] PID ==> Disturbance observer: Yes')
        model.set_disturbance_observer(*get_dobs(kwargs["freq"]), np.array([kwargs["dob"]["Kdob_pos"], kwargs["dob"]["Kdob_neg"]]))
    else:
        print(f'[ INFO] PID ==> Disturbance observer: No')
    
    print(color('[INPUT] PID ==> Feedforward:', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    # is_ff = input(color('\tFeedforward: ', 'blue')) 
    is_ff = '1'
    delete_lines(5)
    if is_anti_windup == "1":
        print(f'[ INFO] PID ==> Feedforward: Yes')
        model.set_feedforward(**kwargs["ff"])
    else:
        print(f'[ INFO] PID ==> Feedforward: No')

    try:
        state, info = env.reset()
        curr_time = 0
        if ref.max_time == float('inf'):
            ref.max_time = 100
        
        infos = deque()
        while curr_time < ref.max_time:
            action = model.predict(state)
            state, _, _, _, info = env.step(action)
            curr_time = info['obs']['curr_time']
            infos.append(info['obs'])

    except KeyboardInterrupt:
        print()
        print('[ INFO] Keyboard interrupt received.')
    
    finally:
        env.step(np.array([1.0, 1.0]))

        datas = infos_to_datas(infos)
        save_datas(datas, model_name, obs_mode, ref_mode, kwargs, save_name)
        plot_datas(datas, save_name)
        env.close()