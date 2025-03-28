import torch
import torch.nn as nn
import numpy as np

class Encode:
    def __init__(self,tensors,num_agents):
        self.tensors = tensors
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = 42
        torch.manual_seed(seed)
        self.encoder = ObservationEncoder().to(self.device)
    def begin(self):
        encoded_channels = {}
        for itr in range(len(self.tensors.keys())):
            # first_channels = self.tensors[itr]['channel 1']
            second_channels = self.tensors[itr]['channel 2']
            third_channels = self.tensors[itr]['channel 3']
            encoded_channels_steps = {}
            
            for step in range(len(second_channels)):
                batch_channels = []
                for i in range(self.num_agents):
                    agent_channels = np.stack([
                        second_channels[step][i],
                        third_channels[step][i]
                    ])
                    batch_channels.append(agent_channels)
                
                batch_tensor = torch.tensor(np.array(batch_channels), dtype=torch.float32).to(self.device)
                
                with torch.no_grad(): 
                    encoded_outputs = self.encoder(batch_tensor) 
                
                encoded_channels_steps[step] = encoded_outputs.cpu().numpy()
            
            encoded_channels[itr] = encoded_channels_steps
        
        return encoded_channels
class ObservationEncoder(nn.Module):
    def __init__(self, input_channels=2, feature_dim=128):  # Changed input_channels to 2
        super().__init__()
        # Block 1: 2x9x9 -> 32x4x4
        seed = 42
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces spatial dim to 4x4
        # Block 2: 32x4x4 -> 64x2x2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dim to 2x2
        # Block 3: 64x2x2 -> 128x2x2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Fully connected layer to produce 128-D feature vector
        self.fc = nn.Linear(128 * 2 * 2, feature_dim)

    def forward(self, x):
        seed = 42
        torch.manual_seed(seed)
        x = self.encoder(x)  # Output: (batch, 32, 4, 4)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128*2*2)
        x = self.fc(x)  # Output: (batch, 128)
        return x