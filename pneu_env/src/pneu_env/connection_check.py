import time
import numpy as np

from pneu_env.real import PneuReal
from pneu_env.src.pneu_env.ref.env_prev import PneuEnv

freq = 50
real = PneuReal(
    freq = freq
)

start_time = time.time()
for _ in range(100):
    flag_time = time.time()
    ctrl = np.array([-1, 1], dtype=np.float64)
    goal = np.array([120, 130], dtype=np.float64)
    obs, info = real.get_obs(ctrl, goal)
    print(list(info['Observation'].values()))
    


