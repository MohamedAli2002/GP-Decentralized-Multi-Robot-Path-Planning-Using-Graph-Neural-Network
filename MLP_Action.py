import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_actions=5):
        super(ActionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)