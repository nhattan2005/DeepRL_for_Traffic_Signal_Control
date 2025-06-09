import torch
import torch.nn as nn
import torch.nn.init as init

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )
        # Khởi tạo trọng số Xavier normal theo paper
        init.xavier_normal_(self.model[0].weight)
        init.xavier_normal_(self.model[2].weight)
        init.xavier_normal_(self.model[4].weight)
        init.zeros_(self.model[0].bias)
        init.zeros_(self.model[2].bias)
        init.zeros_(self.model[4].bias)

    def forward(self, x):
        return self.model(x)