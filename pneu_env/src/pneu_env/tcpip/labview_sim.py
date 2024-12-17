import socket
import struct
import time
import numpy as np

from pneu_env.sim import PneuSim
from pneu_env.src.pneu_env.ref.env_prev import PneuEnv


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 5555))
server.listen()

freq = 200
sim = PneuSim(
    freq = freq,
    delay = 0.02,
    noise = True,
    noise_std = 0.25
)

print('[ INFO] LabView ==> Wait for connection...')
client, client_addr = server.accept()
print(f'[ INFO] LabView ==> Connected ({client_addr})')

while True:
    encoded_ctrl_msg = client.recv(7*4)
    if not encoded_ctrl_msg:
        break
    ctrl_msg = struct.unpack('f'*7, encoded_ctrl_msg)
    ctrl_msg = list(ctrl_msg)

    ###
    # 1. Simulation -> get obs data
    flag_time = time.time()
    ctrl = np.array([
        ctrl_msg[-2],
        ctrl_msg[-1]
    ])
    goal = np.array([
        ctrl_msg[-4],
        ctrl_msg[-3]
    ])
    _, info = sim.get_obs(ctrl, goal)
    # 2. Wait few second -> send obs data
    time.sleep(max(1/freq - time.time() + flag_time, 0))
    # 3. Connection check
    obs_msg = list(info['Observation'].values())
    ###
    
    encoded_obs_msg = struct.pack('f'*7, *obs_msg)
    client.sendall(encoded_obs_msg)
    