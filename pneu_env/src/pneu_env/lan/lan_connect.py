from typing import Dict
import socket
import struct
import json
import time
import threading

from pneu_utils.utils import get_pkg_path

# Configuration
TCP_IP = "127.0.0.1"
TCP_PORT = 5005

FREQ = 50
OBS_FILE = "obs.json"
CTRL_FILE = "ctrl.json"
DATA_KEYS = [
    "time",
    "sen_pos",
    "sen_neg",
    "ref_pos",
    "ref_neg",
    "ctrl_pos",
    "ctrl_neg"
]

def unpack_data(msg: bytes) -> Dict[str, float]:
    decoded_msg = struct.unpack('f'*len(DATA_KEYS), msg)

    data = {}
    for idx, key in enumerate(DATA_KEYS):
        data[key] = decoded_msg[idx]

    return data
    
def pack_data(data: Dict[str, float]) -> bytes:
    msg = list(data.values())
    encoded_msg = struct.pack('f'*len(DATA_KEYS), *msg)
    return encoded_msg

def save_data(
    data: Dict[str, float],
    path: str = OBS_FILE
) -> None:
    with open(path, 'w') as f:
        json.dump(data, f)

def read_data(path: str) -> Dict[str, float]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def receive_message(conn):
    err = True
    while err:
        msg = conn.recv(len(DATA_KEYS)*4)
        err = True if not msg else False
    return msg

def send_message(conn, msg: bytes):
    conn.sendall(msg)
    
def main_loop(conn):
    interval = 1.0/FREQ
    
    pkg_path = get_pkg_path("pneu_env")
    obs_file_path = f"{pkg_path}/src/pneu_env/lan/{OBS_FILE}"
    ctrl_file_path = f"{pkg_path}/src/pneu_env/lan/{CTRL_FILE}"

    while not stop_flag.is_set():
        iter_start_time = time.time()
        
        recv_msg = receive_message(conn)
        recv_data = unpack_data(recv_msg)
        save_data(recv_data, obs_file_path)

        send_data = read_data(ctrl_file_path)
        send_msg = pack_data(send_data)
        send_message(conn, send_msg)
        
        iter_end_time = time.time()
        sleep_time = max(0, interval - iter_start_time + iter_end_time)
        time.sleep(sleep_time)

if __name__ == "__main__":        
    # Initialize socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((TCP_IP, TCP_PORT))
    server_sock.listen(1)

    # Control flag for stopping the loop
    stop_flag = threading.Event()

    try:
        print("[ INFO] Waiting for a connection...")
        conn, addr = server_sock.accept()
        print(f"[ INFO] Connection extablished with {addr}")
        main_loop(conn)
    except KeyboardInterrupt:
        print("[ INFO] Keyboard interrupt received. Stopping the loop.")
        stop_flag.set()
    finally:
        if conn:
            conn.close()
        server_sock.close()
