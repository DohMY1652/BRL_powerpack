import socket
import struct
import time
import threading
import json
import rospy

from pneu_msg.msg import Observation

stop = False
start_time = time.time()

def read_ctrl_file():
    try:
        with open('ctrl.json', 'r') as f:
            ctrl_data = json.load(f)
            print(ctrl_data)
    except:
        with open('ctrl_backup.json', 'r') as f:
            ctrl_data = json.load(f)
            print(ctrl_data)
    
    return list(ctrl_data.values())

def write_obs_file(obs):
    obs_data = dict(
        time = obs[0],
        sen_pos = obs[1],
        sen_neg = obs[2],
        ref_pos = obs[3],
        ref_neg = obs[4],
        ctrl_pos = obs[5],
        ctrl_neg = obs[6]
    )
    with open('obs.json', 'w') as f:
        json.dump(obs_data, f)
    with open('obs_backup.json', 'w') as f:
        json.dump(obs_data, f)

def write_ctrl_file(obs):
    obs_data = dict(
        time = obs[0],
        sen_pos = obs[1],
        sen_neg = obs[2],
        ref_pos = obs[3],
        ref_neg = obs[4],
        ctrl_pos = obs[5],
        ctrl_neg = obs[6]
    )
    with open('ctrl.json', 'w') as f:
        json.dump(obs_data, f)
    with open('ctrl_backup.json', 'w') as f:
        json.dump(obs_data, f)

def send_loop(client_socket):
    global stop
    while not stop:
        try:
            ctrl_data = read_ctrl_file()
            message = struct.pack('f'*7, *ctrl_data)
            client_socket.sendall(message)
        except KeyboardInterrupt:
            stop = True
    
def receive_loop(client_socket):
    global stop
    # pub = rospy.Publisher('/TCP_obs', Observation, queue_size=1)
    while not stop:
        try:
            data = client_socket.recv(28)
            if not data:
                break
            obs_data = struct.unpack('f'*7, data)
            obs_data = list(obs_data)

            # msg = Observation()
            # msg.time = obs_data[0]
            # msg.sen_pos = obs_data[1]
            # msg.sen_neg = obs_data[2]
            # msg.ref_pos = obs_data[3]
            # msg.ref_neg = obs_data[4]
            # msg.ctrl_pos = obs_data[5]
            # msg.ctrl_neg = obs_data[6]
            # pub.publish(msg)
            
            obs_data[5] += 1
            write_obs_file(obs_data)
            # write_ctrl_file(obs_data)
            print(f'Received: {obs_data} Time: {time.time() - start_time}')
        except KeyboardInterrupt:
            stop = True
        
def client_main():
    write_ctrl_file([0, 0, 0, 0, 0, 0, 0])
    global stop
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5555))
    
    try:
        send_thread = threading.Thread(
            target = send_loop,
            args = (client,)
        )
        receive_thread = threading.Thread(
            target = receive_loop,
            args = (client,)
        )

        send_thread.daemon = True
        receive_thread.daemon = True

        send_thread.start()
        receive_thread.start()

        send_thread.join()
        receive_thread.join()

    except KeyboardInterrupt:
        stop = True
    
if __name__ == '__main__':
    # rospy.init_node('TCP_client')
    client_main()