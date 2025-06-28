import socket
import serial
import time
from datetime import datetime
import select
import random
import torch.nn as nn
from pre_processing import Preprocessing
from Encoder import PaperCNN
from GNN_file import PaperGNN
from MLP_Action import PaperMLP
from Adjacency_Matrix import adj_mat
import torch
import numpy as np

actions = [[0,0],[-1,0],[0,1],[1,0],[0,-1]]
GRID_SIZE = 3  
BASE_PORT = 40000
BROADCAST_IP = '255.255.255.255'
FOV_RADIUS = 3
TCP_PORT = 50000
INIT_PORT = 39000

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('dataset_10000_on_10_robots.pth', map_location=device)
r_weight = checkpoint['rank_state_dict']['weight']
num_embeddings, embedding_dim = r_weight.shape
r = nn.Embedding(num_embeddings, embedding_dim).to(device=device)
cnn = PaperCNN(2).to(device=device)
gnn = PaperGNN(input_dim=160,output_dim=160).to(device=device)
mlp = PaperMLP(input_dim=160,output_dim=5).to(device=device)
r.load_state_dict(checkpoint['rank_state_dict'])
cnn.load_state_dict(checkpoint['cnn_state_dict'])
gnn.load_state_dict(checkpoint['gnn_state_dict'])
mlp.load_state_dict(checkpoint['mlp_state_dict'])
r.eval()
cnn.eval()
gnn.eval()
mlp.eval()

def prepare_data(tensors,num_of_robots,cases,rfov):
    dataset = []
    adj_obj = adj_mat(cases,rfov)
    adj_cases = adj_obj.get_adj_mat()
    for itr in range(len(tensors.keys())):
        second_channels = tensors[itr]['channel 2']
        third_channels = tensors[itr]['channel 3']
        for step in range(len(second_channels)):
            batch_channels = []
            for i in range(num_of_robots):
                agent_channels = np.stack([
                    second_channels[step][i],
                    third_channels[step][i]
                ])
                batch_channels.append(agent_channels)
            batch_tensor = torch.tensor(np.array(batch_channels), dtype=torch.float32)
            adj_step = torch.tensor(np.array(adj_cases[itr][step]),dtype=torch.float32)
            dataset.append((batch_tensor,adj_step))
    return dataset

def field_of_view_nodes(i, j, r_fov):
    nodes = []
    for k in range(i - r_fov, i + r_fov + 1):
        for l in range(j - r_fov, j + r_fov + 1):
            if (0 <= k < GRID_SIZE and 0 <= l < GRID_SIZE and (i, j) != (k, l)):
                nodes.append((k, l))
    return nodes

def bind_udp_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.setblocking(False)
    time.sleep(1)
    return s

def node_to_port(x, y):
    return BASE_PORT + (x * GRID_SIZE + y)

def start_tcp_listener(my_ip, data_received_callback):
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_sock.bind((my_ip, TCP_PORT))
    tcp_sock.listen(10)
    tcp_sock.setblocking(False)
    return tcp_sock

def network_process(robot_id, ip_address, x, y, x_g , y_g):
    tcp_listener = start_tcp_listener(ip_address, None)
    state = 0
    while True:
        if x == x_g and y == y_g :
             print("✅ Robot 1 reached goal successfully")
             break  

        current_udp_port = node_to_port(x, y)
        udp_sock = bind_udp_port(current_udp_port)
        received_ips = set()
        received_model_data = []

        # Step 1: Broadcast IP to ALL Fov nodes 
        fov_nodes = field_of_view_nodes(x, y, FOV_RADIUS)
        message = ip_address.encode()
        for nx, ny in fov_nodes:
            target_port = node_to_port(nx, ny)
            for _ in range(3):
                udp_sock.sendto(message,('255.255.255.255', target_port))
            print(f"[Step 1] Robot {robot_id} broadcasted IP to ({nx},{ny}) on port {target_port}")

        # Step 2: Listen for incoming IPs
        print(f"[Step 2] Robot {robot_id} listening for IPs on port {current_udp_port}...")
        end_time = time.time() + 2
        while time.time() < end_time:
            ready, _, _ = select.select([udp_sock], [], [], 2.0)
            if udp_sock in ready:
                try:
                    data, addr = udp_sock.recvfrom(1024)
                    sender_ip = data.decode()
                    print(f"[Step 2] Received IP from {addr}: {sender_ip}")
                    received_ips.add(sender_ip)
                except Exception:
                    pass

        step_case = {"start_positions": [[x,y]], "goal_positions": [[x_g,y_g]],"paths":[[[x,y]]]}
        grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] 
        p = Preprocessing(grid,[step_case],FOV_RADIUS)
        data_list = p.begin()
        data_list = prepare_data(data_list,1,[step_case],3)
        data_list = torch.tensor((data_list[0][0]),dtype=torch.float32)
        data_list = data_list.view(1,2,9,9)
        e = cnn(data_list)
        ranks = torch.tensor([robot_id])
        rank = r(ranks)
        cnn_with_ranks = torch.cat([e,rank],dim=-1)
        data_list = list(cnn_with_ranks[0])

        # Step 3: Send model data to LIVE Fov nodes only
        data_message = ','.join([str(val.item()) for val in data_list])
        for ip in received_ips:
            try:
                tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                tcp_client.settimeout(1)
                tcp_client.connect((ip, TCP_PORT))
                tcp_client.sendall(data_message.encode())
                tcp_client.close()
                print(f"[Step 3] Sent model data to {ip}")
            except Exception as e:
                print(f"[Step 3] Failed to send to {ip}: {e}")
        time.sleep(2)
        received_model_data.append(data_list)

        # Step 4: Listen for model data 
        print(f"[Step 4] Listening for TCP data...")
        end_time = time.time() + 2
        while time.time() < end_time:
            ready, _, _ = select.select([tcp_listener], [], [], 2.5)
            if tcp_listener in ready:
                conn, addr = tcp_listener.accept()
                try:
                    data = b''
                    while True:
                        part = conn.recv(4096)
                        if not part:
                            break
                        data += part
                        if len(data) >= 160:
                            break
                    data = data.decode()
                    model_values = list(map(float, data.split(',')))
                    received_model_data.append( model_values)
                    print(f"[Step 4] Received model data from {addr}")
                except Exception as e:
                    print(f"[Step 4] Error decoding data: {e}")
                finally:
                    conn.close()

        new_comm_gnn = gnn(torch.tensor(received_model_data,dtype=torch.float32),torch.ones(len(received_model_data),len(received_model_data))).view(1,len(received_model_data),160)

        frame_0 = ['i','l','f','r','b']
        frame_1 = ['i','f','r','b','l']
        frame_2 = ['i','r','b','l','f']
        frame_3 = ['i','b','l','f','r']
        frames = [frame_0,frame_1,frame_2,frame_3]

        for robot in range(1):
            prediction = mlp(torch.tensor(new_comm_gnn[0][robot],dtype=torch.float32).unsqueeze(0).to(device))
            predicted_class = torch.argmax(prediction, dim=1)
            movement = frames[state][predicted_class]
            if frames[state][predicted_class] == 'l':
                state = (state + 1) % 4
            elif frames[state][predicted_class] == 'r':
                state = (state - 1) % 4
            ser = serial.Serial('/dev/ttyUSB0',115200,timeout=1,bytesize=serial.EIGHTBITS,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE)
            time.sleep(0.3)  
            ser.write(movement.encode())
            x = x+actions[predicted_class][0]
            y = y+actions[predicted_class][1]
        time.sleep(0.1)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    robot_id = 1 
    ip_address = get_local_ip()
    x_s = 1
    y_s = 0
    x_g = 2
    y_g = 2
    while True:
        now = datetime.now()
        if (now.hour == 11 and 
            now.minute == 24 and 
            now.second == 00):
            print(f"[INFO] Time matched! Starting at {now.strftime('%H:%M:%S')}")
            break
    network_process(robot_id, ip_address, x_s, y_s, x_g, y_g)
