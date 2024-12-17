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
from pneu_env.env import PneuEnv
from pneu_env.sim import PneuSim
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

if __name__ == '__main__':
    try:
        print('[ INFO] Control Mode: RL')
        print(color('[INPUT] Control Model:', 'blue'))
        
        models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
        for i, model in enumerate(models):
            print(color(f'\t{i+1}. {model}', 'yellow'))
        print(color('\t---', 'blue'))
        model_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
        model_name = models[model_idx]
        delete_lines(len(models) + 3)
        print(f'[ INFO] Control Model: {model_name}')

        kwargs = load_yaml(model_name)
            
        print(color('[INPUT] Reference Mode:', 'blue'))
        print(color('\t1. Step case', 'yellow'))
        print(color('\t2. Random', 'yellow'))
        print(color('\t3. Step positive', 'yellow'))
        print(color('\t4. Sinusoidal', 'yellow'))
        print(color('\t---', 'blue'))
        ref_mode = input(color('\tREF: ', 'blue')) 
        delete_lines(7)

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
            print(f'[ INFO] Reference Mode: Step positive')
            ref = StepRef(
                time_step = 10,
                ref_pos = [110, 130, 130, 110, 130, 130, 110],
                ref_neg = [60, 60, 60, 60, 60, 60, 60],
            )
        elif ref_mode == '4':
            print(f'[ INFO] Reference Mode: Sinusoidal')
            ref = SineRef(
                pos_amp = 10,
                pos_per = 4,
                pos_off = 170,
                neg_amp = 10,
                neg_per = 5,
                neg_off = 50,
                iter = 2
            )

        print(color('[INPUT] Observation Mode:', 'blue'))
        print(color('\t1. Sim', 'yellow'))
        print(color('\t2. Real', 'yellow'))
        print(color('\t---', 'blue'))
        obs_mode = input(color('\tOBS: ', 'blue')) 
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
        save_name = f'{formatted_time}_{model_name}' if data_log == '1' else None
        print(f'[ INFO] Data logging: {"False" if data_log == "2" else save_name}.csv')

        if obs_mode == '1':    
            obs = PneuSim(
                freq = 50,
                delay = 0.1,
                noise = False,
                noise_std = 0,
                offset_pos = 0,
                offset_neg = 0,
                scale = True
            )
            pred = PneuPred(
                freq = 50,
                delay = 0,
                noise = False,
                scale = True,
            )
        elif obs_mode == '2':
            obs = PneuReal(
                freq = 50,
                scale = True
            )
            pred = PneuPred(
                freq = 50,
                delay = 0,
                noise = False,
                scale = True
            )

        obs.set_init_press(
            init_pos_press = 170,
            init_neg_press = 50
        )

        env = PneuEnv(
            obs = obs,
            ref = ref,
            pred = pred if kwargs['pred'] is not None else None,
            **kwargs['env']
        )

        model = SAC(
            env = env,
            **kwargs['model']
        )
        model.load(name = model_name)

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

        env.close()
        
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

        if save_name is not None:
            os.makedirs(f'{get_pkg_path("pneu_rl")}/exp/{save_name}')
            df = pd.DataFrame(datas)
            df.to_csv(f'/Users/greenlandshark/MATLAB/csv/datas/{save_name}.csv', index=False)
            df.to_csv(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/{save_name}.csv', index=False)
    
        config = dict(
            model_name = model_name,
            obs_mode = "Simulation" if obs_mode == "1" else "Real",
            ref_mode = ref_mode
        )
        if save_name is not None:
            with open(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/cfg.yaml', 'w') as f:
                yaml.dump(config, f)
    
        
        fontname = 'Times New Roman'
        label_font_size = 18
        fig_name = 'fig'
        
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
            plt.savefig(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/{save_name}.png')
        plt.show()
    
    except KeyboardInterrupt:
        exit()
            
