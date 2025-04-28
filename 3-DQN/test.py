import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_channel, width, action_nums):
        super().__init__()
        self.flatten_size = input_channel*width**2
        self.q_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_nums)
        )
    def forward(self, x):
        q_values = self.q_network(x)
        return q_values
    
q = QNetwork(1,10,4)
dummy_input = torch.rand(3,1,10,10)
output = q(dummy_input)
print(output, output.shape)