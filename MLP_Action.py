import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_actions=4):
        super(ActionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        # x is the feature vector from the GNN for each robot
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Output a probability distribution over actions
        action_prob = F.softmax(x, dim=-1)
        return action_prob