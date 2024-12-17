import os

from collections import deque
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
import pickle
import yaml

from pneu_ref.random_ref import RandomRef
from pneu_ref.step_ref import StepCasesRef, StepRef
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
    print('[ INFO] Control Mode: RL')
    print(color('[INPUT] Control Model:', 'blue'))
    
    models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
    for i, model in enumerate(models):
        print(color(f'\t{i+1}. {model}', 'yellow'))
    print(color('\t---', 'blue'))
    model_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
    model_name = models[model_idx]
    delete_lines(len(models) + 3)
    print(f'[ INFO] Model: {model_name}')

    print(color('[INPUT] Info type:', 'blue'))
    print(color('\t1. Rewards', 'yellow'))
    print(color('\t2. Alphas', 'yellow'))
    print(color('\t3. Temporal weight', 'yellow'))
    info_mode = input(color('\tTYPE: ', 'blue')) 
    delete_lines(5)
    
    if info_mode == '1':
        print(f'[ INFO] Info type: Rewards')
        info_type = 'reward'
    elif info_mode == '2':
        print(f'[ INFO] Info type: Alphas')
        info_type = 'alpha'
    elif info_mode == '3':
        print(f'[ INFO] Info type: Temporal weight')
        info_type = 'temporal'
    
    infos_file = f'{get_pkg_path("pneu_rl")}/models/{model_name}/infos.pkl'
    with open(infos_file, 'rb') as f:
        infos = pickle.load(f)
    
    epi = deque()
    step = deque()
    reward = deque()
    alpha = deque()
    temporal_weight = deque()
    for k, v in infos.items():
        epi.append(k)
        step.append(v['steps'])
        reward.append(v['reward'])
        if isinstance(v['alpha'], torch.Tensor):
            alpha.append(v['alpha'].detach().item())
        else:
            alpha.append(v['alpha'])
        temporal_weight.append(v['temporal_weight'])

    data = dict(
        epi = np.array(epi),
        step = np.array(step),
        reward = np.array(reward),
        alpha = np.array(alpha),
        temporal = np.array(temporal_weight)
    )

    df = pd.DataFrame(data)
    df.to_csv('info.csv', index=False)
    
    fontname = 'Times New Roman'
    label_font_size = 18
    fig_name = 'fig'
    
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0,0])

    ax1.plot(
        np.array(data['epi']),
        np.array(data[info_type]),
        linewidth=2, color='black'
    )
    ax1.set_xlabel('Step', fontname=fontname, fontsize=label_font_size)
    if info_type == 'alpha':
        ax1.set_ylabel(r'Temperature parameter ($\alpha$)', fontname=fontname, fontsize=label_font_size)
    elif info_type == 'reward':
        ax1.set_ylabel('Reward', fontname=fontname, fontsize=label_font_size)
    elif info_type == 'temporal':
        ax1.set_ylabel('Temporal weight', fontname=fontname, fontsize=label_font_size)
    ax1.grid(which='major', color='silver', linewidth=1)
    ax1.grid(which='minor', color='lightgray', linewidth=0.5)
    ax1.minorticks_on()
    ax1.legend(loc='upper right')
    ax1.set(xlim=(None, None), ylim=(None, None))

    plt.tight_layout()
    # if save_name is not None:
    #     plt.savefig(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/{save_name}.png')
    plt.show()
        
