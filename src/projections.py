import torch
import torch.nn as nn
import torch.nn.functional as F


class proj_v1(nn.Module):
    def __init__(self, in_dim):
        super(proj_v1, self).__init__()
        self.in_dim = in_dim
        hidden_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    # def forward(self, z1, z2):
    #     z1 = self.fc2(F.relu(self.fc1(z1)))
    #     z2 = self.fc2(F.relu(self.fc1(z2)))
    #     z1 = F.normalize(z1, dim=1)
    #     # print(torch.norm(g1[0]))
    #     z2 = F.normalize(z2, dim=1)

    #     return z1, z2
    def forward(self, z):
        z = self.fc2(F.relu(self.fc1(z)))

        return z

