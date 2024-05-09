import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F

from mlp import MLP

class NeuroDiver(nn.Module):
    def __init__(self, args):
        super(NeuroDiver, self).__init__()
        self.embedding_dim = args.dim
        self.depth = args.n_rounds
        self.weight_share = args.weight_share
        self.residual = args.residual
        self.projection = [MLP(self.embedding_dim, self.embedding_dim, self.embedding_dim)]
        if self.weight_share:
            self.projection.append(MLP(self.embedding_dim * 2, int(self.embedding_dim * 1.5), self.embedding_dim))
        else:
            for _ in range(self.depth - 1):
                if self.residual:
                    self.projection.append(MLP(self.embedding_dim, self.embedding_dim, self.embedding_dim))
                else:
                    self.projection.append(MLP(self.embedding_dim * 2, int(self.embedding_dim * 1.5), self.embedding_dim))
        self.projection = nn.ModuleList(self.projection)
        self.layernorm = LayerNorm(self.embedding_dim)
        #self.layernorm2 = LayerNorm(self.embedding_dim*2)


    def forward(self, problem):
        n_nodes = problem.n_node
        #print("total noides")
        #print(n_nodes)
        n_offset = problem.node_offset
        #print("offset")
        #print(n_offset)
        n_edge = problem.n_edge
        n_probs = len(n_offset)

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()

        #print("adjacency matrix")
        #print(ts_L_unpack_indices)

        L_unpack = (torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(n_edge), torch.Size([n_nodes, n_nodes])).to_dense() + torch.eye(n_nodes)).cuda()

        #L_unpack = (torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(n_edge), torch.Size([n_nodes, n_nodes])).to_dense()).cuda()
        #print(degree)
        #print(L_unpack)

        degree = torch.count_nonzero(L_unpack, dim = 0)

        #print(torch.sort(degree))

        new_embedding = F.one_hot(degree, num_classes = self.embedding_dim).float().cuda()

        #new_embedding = torch.ones((n_nodes, self.embedding_dim)).cuda()

        #new_embedding = torch.rand(n_nodes, self.embedding_dim).cuda()

        old_embedding = new_embedding.cuda()
        if not self.residual:
            for i in range(self.depth):
                if i == 0:
                    old_embedding = new_embedding
                else:
                    old_embedding = new_embedding[:, self.embedding_dim:]
                if self.weight_share:
                    if i == 0:
                        new_embedding = torch.matmul(L_unpack, self.projection[0](new_embedding))
                    else:
                        new_embedding = torch.matmul(L_unpack, self.projection[1](new_embedding))
                else:
                    new_embedding = torch.matmul(L_unpack, self.projection[i](new_embedding))
                new_embedding = self.layernorm(new_embedding)
                if i != self.depth - 1:
                    new_embedding = torch.cat((old_embedding, new_embedding), 1)
        else:
            for i in range(self.depth):
                old_embedding = new_embedding.clone()
                new_embedding = torch.matmul(L_unpack, self.projection[i](new_embedding))
                new_embedding = self.layernorm(new_embedding)
                new_embedding = (new_embedding + old_embedding) * 0.5

        representation = torch.zeros((n_probs, self.embedding_dim)).cuda()

        for i in range(n_probs):
            if i == n_probs-1:
                representation[i] = torch.mean(new_embedding[n_offset[i]:], dim = 0)
            else:
                #print(torch.mean(new_embedding[n_offset[i]:n_offset[i+1], 0]))
                representation[i] = torch.mean(new_embedding[n_offset[i]:n_offset[i+1]], dim = 0)
                #print(representation[i][0])
                #print(" ")
        print(representation)
        return representation








