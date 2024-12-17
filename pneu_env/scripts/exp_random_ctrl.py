import numpy as np
import pandas as pd
from collections import deque

from datetime import datetime
import time

from pneu_env.real import PneuReal
from pneu_env.sim import PneuSim
from pneu_ref.ctrl_ref import CtrlRandom
from pneu_utils.utils import get_pkg_path


ATM = 101.325
dummy_goal = np.array([ATM, 0], dtype=np.float64)

tag = input("File tag: ")
tag = f"_{tag}" if len(tag) != 0 else ""

matlab_path = "/Users/greenlandshark/MATLAB/main/datas"
# matlab_path = "/Users/greenlandshark/MATLAB/main/remodeling/data"

now = datetime.now()
formatted_time = now.strftime("%y%m%d_%H_%M_%S")
save_file_name = f"{formatted_time}_Flowrate{tag}"


env = PneuReal(freq = 50)
# env = PneuSim(freq = 50)
ctrl = CtrlRandom()

data = dict(
    curr_time = deque(),
    press_in = deque(),
    press_out = deque(),
    flowrate1 = deque(),
    flowrate2 = deque(),
    ctrl_pos = deque(),
    ctrl_neg = deque(),
    press_pos = deque(),
    press_neg = deque(),
)

try:
    if ctrl.max_time == float('inf'):
        ctrl.max_time = 600

    curr_time = 0
    while curr_time < ctrl.max_time:
        curr_ctrl = ctrl.get_ctrl(curr_time)
        # curr_ctrl[0] = 1
        # curr_ctrl[1] = 1
        obs, info = env.observe(curr_ctrl, dummy_goal)

        print(info["message"])
        
        data["curr_time"].append(obs[0])
        data["press_in"].append(info["message"]["msg1"])
        data["press_out"].append(info["message"]["msg2"])
        data["flowrate1"].append(info["message"]["msg3"])
        data["flowrate2"].append(info["message"]["msg4"])
        data["ctrl_pos"].append(info["message"]["msg5"])
        data["ctrl_neg"].append(info["message"]["msg6"])
        data["press_pos"].append(info["message"]["msg1"])
        data["press_neg"].append(info["message"]["msg2"])

        curr_time = obs[0]
        
except KeyboardInterrupt:
    pass

finally:
    ctrl = np.array([1.0, 1.0], dtype=np.float64)
    _, _ = env.observe(ctrl, dummy_goal)

    for k, v in data.items():
        data[k] = np.array(v, dtype=np.float64)

    df = pd.DataFrame(data)
    df.to_csv(f"{matlab_path}/{save_file_name}.csv", index=False)
    df.to_csv(f"{get_pkg_path('pneu_env')}/exp/{save_file_name}.csv", index=False)