import numpy as np
import torch
import torch.nn as nn
import random
from Adjacency_Matrix import adj_mat
from Dataset_Generator import DatasetGenerator
from pre_processing import Preprocessing
from Encoder import PaperCNN
from GNN_file import PaperGNN
from MLP_Action import PaperMLP
class TestModel:
    def __init__(self,num_of_robots, num_of_cases, grid_size, model_pth = 'checkpoint.pth', random_seed = 18, rfov = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_of_robots = num_of_robots
        self.rfov = rfov
        self.grid_size = grid_size
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        new_dataset_generator = DatasetGenerator(num_cases=num_of_cases, num_agents=self.num_of_robots, grid=self.grid)
        self.eval_cases = new_dataset_generator.generate_cases()
        
        checkpoint = torch.load(model_pth, map_location=self.device)
        r_weight = checkpoint['rank_state_dict']['weight']
        num_embeddings, embedding_dim = r_weight.shape
        
        self.r = nn.Embedding(num_embeddings, embedding_dim).to(self.device)
        self.cnn = PaperCNN(2).to(device=self.device)
        self.gnn = PaperGNN(input_dim=160,output_dim=160).to(device=self.device)
        self.mlp = PaperMLP(input_dim=160,output_dim=5).to(device=self.device)
        
        self.r.load_state_dict(checkpoint['rank_state_dict'])
        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        
        self.r.eval()
        self.cnn.eval()
        self.gnn.eval()
        self.mlp.eval()
        
    def is_swaped(self,original, new):
        for o_itr,(i,j) in enumerate(original):
            for n_itr,(k,l) in enumerate(new):
                if (i == k and j == l):
                    if(new[o_itr][0] == original[n_itr][0] and new[o_itr][1] == original[n_itr][1] and o_itr != n_itr):
                        return True
        return False
    
    def prepare_data(self,tensors,num_of_robots,cases):
        dataset = []
        adj_obj = adj_mat(cases,self.rfov)
        adj_cases = adj_obj.get_adj_mat()
        for itr in range(len(tensors.keys())):
            # first_channels = self.tensors[itr]['channel 1']
            second_channels = tensors[itr]['channel 2']
            third_channels = tensors[itr]['channel 3']
            random.seed(itr)
            for step in range(len(second_channels)):
                batch_channels = []
                # ranks = set()
                # i = 0
                # while(i<num_of_robots):
                # # for i in range(num_of_robots):
                #   rank_feature = random.randint(1,254)
                #   ranks.add(float(rank_feature)/255)
                #   if len(ranks) != i+1:
                #       i-=1
                #   i+=1
                # # print(list(bb)[0])
                # rank_list = list(ranks)
                # rank_list.sort(reverse=True)
                rank_list = [10,9,8,7,6, 5, 4, 3, 2, 1]
                for i in range(num_of_robots):
                    agent_channels = np.stack([
                        second_channels[step][i],
                        third_channels[step][i]
                    ])
                    batch_channels.append(agent_channels)
                batch_tensor = torch.tensor(np.array(batch_channels), dtype=torch.float32)
                adj_step = torch.tensor(np.array(adj_cases[itr][step]),dtype=torch.float32)
                ranks_tensor =torch.tensor(rank_list,dtype=torch.long)
                dataset.append((batch_tensor,adj_step,ranks_tensor))
        return dataset
    
    def test(self):
        all_confilicts = 0
        all_steps = 0
        confilict_cases = 0
        conflict_set_cases = set()
        ranks = [i for i in range(self.num_of_robots)]
        ranks.reverse()
        ranks = torch.tensor(ranks).to(self.device)
        reached = 0
        actions = [[0,0],[-1,0],[0,1],[1,0],[0,-1]]
        with torch.inference_mode():
            for itr_case, case in enumerate(self.eval_cases):
                threshold = 0
                for path in case['paths']:
                    threshold+= len(path)
                threshold = 3*threshold # from the paper
                num_of_steps = 0
                robots_reached = set()
                step_number = 0
                previous_nodes = [0 for i in range(self.num_of_robots)]
                current_nodes = []
                path_robot_zero = []
                num_of_confilicts = 0
                flag_case = 0
                while num_of_steps <= threshold and len(robots_reached) <self.num_of_robots:
                    paths = []
                    goals = []
                    for robot in range(self.num_of_robots):
                        if step_number == 0:
                            current_node = case['paths'][robot][step_number]
                            current_nodes.append(list(current_node))
                            if robot == 0:
                                # print(current_node)
                                # print(list(case['paths'][robot][-1]))
                                pass
                            paths.append([list(current_node)])
                            goals.append(list(case['goal_positions'][robot]))
                        else:
                            paths.append([current_nodes[robot]])
                            goals.append(list(case['goal_positions'][robot]))
                    # print(current_nodes[0])
                    step_case = {"start_positions": current_nodes, "goal_positions": goals,"paths":paths}
                    # print(paths)
                    # print("start_preprocessing")
                    unique_nodes = set()
                    for node in current_nodes:
                        unique_nodes.add((node[0],node[1]))
                    num_of_confilicts+= (len(current_nodes)-len(unique_nodes))
                    if ((len(current_nodes)-len(unique_nodes))!=0 and flag_case == 0):
                        conflict_set_cases.add(itr_case)
                    # confilict_cases +=1
                        flag_case = 1
                    p_new = Preprocessing(self.grid,[step_case],self.rfov)
                    data_tensors_new  = p_new.begin()
                    dataset = self.prepare_data(data_tensors_new,self.num_of_robots,[step_case])
                    x = torch.tensor(np.array([tensor for tensor,adj_,rank_i in dataset]),dtype=torch.float32).to(self.device)
                    x = x.view(self.num_of_robots,2,9,9)
                    # print(np.array(x).shape)
                    new_encoder = self.cnn(x).to(self.device)
                    rank = self.r(ranks)
                    cnn_with_ranks = torch.cat([new_encoder,rank],dim=-1)
                    new_comm_gnn = self.gnn(cnn_with_ranks,torch.tensor(np.array([adj_ for tensor,adj_,rank_i in dataset]),device=self.device).view(self.num_of_robots,self.num_of_robots)).view(1,self.num_of_robots,160)
                    # print(new_comm_gnn.shape)
                    future_nodes = list(range(0,self.num_of_robots))
                    for itr_node, c_node in enumerate(current_nodes):
                        previous_nodes[itr_node] = (c_node[0],c_node[1])
                    for robot in range(self.num_of_robots):
                        if robot not in robots_reached:
                            # prediction = model.model(torch.tensor(reshaped_gnn_features[0][robot],dtype=torch.float32).unsqueeze(0).to(device))
                            prediction = self.mlp(torch.tensor(new_comm_gnn[0][robot],dtype=torch.float32).unsqueeze(0).to(self.device))
                            # print(prediction)
                            predicted_class = torch.argmax(prediction, dim=1)
                            right_state = False
                            next = [current_nodes[robot][0] + actions[predicted_class][0],current_nodes[robot][1] + actions[predicted_class][1]]
                            # future_nodes[robot] = (next[0],next[1])
                            if (next[0]>=self.grid_size or next[0]<0 or next[1]>=self.grid_size or next[1]<0):
                                # current_nodes[robot] = [current_nodes[robot][0] + actions[0][0],current_nodes[robot][1] + actions[0][1]]
                                pass
                            else:
                                current_nodes[robot] = [current_nodes[robot][0] + actions[predicted_class][0],current_nodes[robot][1] + actions[predicted_class][1]]
                            if (current_nodes[robot][0] == goals[robot][0]) and (current_nodes[robot][1] == goals[robot][1]):
                                robots_reached.add(robot)
                                # print("trueeeeee")
                    if self.is_swaped(previous_nodes,current_nodes):
                        conflict_set_cases.add(itr_case)
                        num_of_confilicts+=1
                    step_number = 1
                    num_of_steps+=1
                if len(robots_reached) == self.num_of_robots:
                    reached += 1
                # print(f"num of confilicts = {num_of_confilicts}")
                all_confilicts+= num_of_confilicts
                all_steps+= num_of_steps
        print(f"reached cases = {reached}")
        print(f"all confilicts = {all_confilicts}")
        print(f"all steps = {all_steps}")
        print(f"cases with confilict = {len(conflict_set_cases)}")
        return reached, all_confilicts, all_steps, conflict_set_cases
