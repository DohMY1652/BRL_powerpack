import numpy as np
import pandas as pd
from collections import deque

from pneu_ref.traj_ref import TrajRef
from pneu_env.env import PneuEnv
from pneu_env.real import PneuReal

import matplotlib.pyplot as plt

traj_ctrl = np.r_[np.arange(1.0, 0.89, -0.01), np.arange(0.9, 1.01, 0.01)]
traj_time = 5*np.arange(1, len(traj_ctrl) + 1)
traj_no_ctrl = np.ones(traj_ctrl.shape)

collect_mode = input("1. pos, 2. neg : ")
if collect_mode == "1":
    mode = "pos"
    ff_ctrl = TrajRef(
        traj_time = traj_time,
        traj_pos = traj_ctrl,
        traj_neg = traj_no_ctrl
    )
else:
    mode = "neg"
    ff_ctrl = TrajRef(
        traj_time = traj_time,
        traj_pos = traj_no_ctrl,
        traj_neg = traj_ctrl
    )

obs = PneuReal(freq = 50)
dummy_ref = TrajRef(
    traj_time = traj_time,
    traj_pos = 101.325*np.ones(traj_time.shape),
    traj_neg = 101.325*np.ones(traj_time.shape)
)

logger = deque()

curr_time = 0
while curr_time < traj_time[-1]:
    curr_ctrl = ff_ctrl.get_goal(curr_time)
    curr_goal = dummy_ref.get_goal(curr_time)
    curr_obs, _ = obs.get_obs(2*curr_ctrl - 1, curr_goal)
    logger.append(np.r_[curr_obs, curr_ctrl])
    curr_time = curr_obs[0]

keys = ["curr_time", "sen_pos", "sen_neg", "ctrl_pos", "ctrl_neg"]
data = dict()
for k, v in zip(keys, np.stack(logger, axis=1)):
    data[k] = v

df = pd.DataFrame(data)
df.to_csv(f"{mode}.csv", index=False)

is_plot = input("Plot? [y/n] ")
if is_plot == "y":
    plt.figure()
    plt.plot(data["curr_time"], data["sen_pos"])
    plt.show()