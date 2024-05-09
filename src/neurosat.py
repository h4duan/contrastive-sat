import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch_scatter import scatter
from mlp import MLP

class NeuroSAT(nn.Module):
  def __init__(self, args):
    super(NeuroSAT, self).__init__()
    self.args = args

    self.init_ts = torch.ones(1)
    self.init_ts.requires_grad = False

    self.L_init = nn.Linear(1, args.dim)
    self.C_init = nn.Linear(1, args.dim)

    self.L_msg = MLP(self.args.dim, self.args.dim, self.args.dim)
    self.C_msg = MLP(self.args.dim, self.args.dim, self.args.dim)

    #self.L_msg_lm = BatchNorm(self.args.dim, element_wise = False)
    #self.C_msg_lm = BatchNorm(self.args.dim)

    self.L_update = nn.LSTM(self.args.dim*2, self.args.dim)
    # self.L_norm   = nn.LayerNorm(self.args.dim)
    self.C_update = nn.LSTM(self.args.dim, self.args.dim)
    # self.C_norm   = nn.LayerNorm(self.args.dim)
    self.C_lm_1 = LayerNorm(self.args.dim, elementwise_affine = True)
    self.C_lm_2 = LayerNorm(self.args.dim, elementwise_affine = True)
    #self.C_update_lm_3 = LayerNorm(self.args.dim, elementwise_affine = True)
    self.L_lm_1 = LayerNorm(self.args.dim, elementwise_affine = True)
    self.L_lm_2 = LayerNorm(self.args.dim, elementwise_affine = True)

    #self.L_vote = MLP(self.args.dim, self.args.dim, self.args.dim)
    

    self.denom = torch.sqrt(torch.Tensor([self.args.dim]))

  def forward(self, problem):
    n_vars    = problem.n_vars
    n_lits    = problem.n_lits
    n_clauses = problem.n_clauses
    n_probs   = len(problem.offset)
    n_offset = problem.offset
    n_clause_offset = problem.clause_offset

    literal_index = torch.repeat_interleave(torch.arange(n_probs).cuda(), torch.tensor(n_offset).cuda())
    literal_index = torch.reshape(literal_index, (n_vars, 1))
    literal_index = torch.cat((literal_index, literal_index), 0)

    """
    literal_index = torch.zeros((n_vars, 1)).cuda().long()
    #print(n_offset)
    n = n_offset[0]
    for i in range(1, n_probs):
      if i != n_probs-1:
        literal_index[n:n+n_offset[i], :] += i
        #print(literal_index[n:n_offset[i+1], :])
        n += n_offset[i]
      else:
        literal_index[n:, :] += i
    #print(torch.unique(literal_index))
    literal_index = torch.cat((literal_index, literal_index), 0)
    """
    #print(literal_index)
    #print(n_vars, n_lits, n_clauses, n_probs)

    ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()
    
    init_ts = self.init_ts.cuda()
    # 1 x n_lits x dim & 1 x n_clauses x dim
    L_init = self.L_init(init_ts).view(1, 1, -1)
    # print(L_init.shape)
    L_init = L_init.repeat(1, n_lits, 1)
    C_init = self.C_init(init_ts).view(1, 1, -1)
    # print(C_init.shape)
    C_init = C_init.repeat(1, n_clauses, 1)

    # print(L_init.shape, C_init.shape)

    L_state = (L_init, torch.zeros(1, n_lits, self.args.dim).cuda())
    C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
    #L_unpack  = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells), torch.Size([n_lits, n_clauses])).to_dense().cuda()
    L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),torch.Size([n_lits, n_clauses])).cuda()
    #print(L_unpack)
    # print(ts_L_unpack_indices.shape)

    for _ in range(self.args.n_rounds):
      # n_lits x dim
      L_hidden = L_state[0].squeeze(0)
      L_pre_msg = self.L_msg(L_hidden)
      # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
      #LC_msg = torch.matmul(L_unpack.t(), L_pre_msg)
      #print(L_unpack)
      #LC_msg = torch.matmul(L_unpack.t(), L_pre_msg)
      LC_msg = torch.sparse.mm(L_unpack.t(), L_pre_msg)
      # print(L_hidden.shape, L_pre_msg.shape, LC_msg.shape)

      _, C_state= self.C_update(LC_msg.unsqueeze(0), C_state)
      C_state = (self.C_lm_1(C_state[0]), self.C_lm_2(C_state[1]))
      # print('C_state',C_state[0].shape, C_state[1].shape)

      # n_clauses x dim
      C_hidden = C_state[0].squeeze(0)
      C_pre_msg = self.C_msg(C_hidden)
      # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
      CL_msg = torch.sparse.mm(L_unpack, C_pre_msg)
      # print(C_hidden.shape, C_pre_msg.shape, CL_msg.shape)

      _, L_state= self.L_update(torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)
      L_state = (self.L_lm_1(L_state[0]), self.L_lm_2(L_state[1]))
      # print('L_state',C_state[0].shape, C_state[1].shape)

    logits = L_state[0].squeeze(0)
    #print(logits)
    #rep_mean = scatter(logits, literal_index, dim=0, reduce="mean")
    #clauses = C_state[0].squeeze(0) 
    """
    print("logits", logits.shape)
    vote = self.L_vote(logits)
    print('vote', vote.shape)
    vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
    print('vote_join', vote_join.shape)
    self.vote = vote_join
    vote_join = vote_join.view(n_probs, -1, 2).view(n_probs, -1)
    print("vote_join", vote_join.shape)
    vote_mean = torch.mean(vote_join, dim=1)
    print('mean', vote_mean.shape)
    """
    #rep_join = torch.cat([logits[:n_vars, :], logits[n_vars:, :]], dim=1)
    #rep_join_chunk = list(torch.split(rep_join, n_offset))
    #rep_join_mean = torch.zeros((n_probs, self.args.dim * 2)).to(torch.device("cuda"))
    #for i in range(n_probs):
    #    rep_join_mean[i] = torch.mean(rep_join_chunk[i], dim = 0)
    #rep_mean = torch.mean(torch.stack(torch.split(rep_join_mean, self.args.dim, dim = 1)), dim = 0)
    #print(logits)
    #print(rep_mean)
    rep_mean = scatter(logits, literal_index, dim=0, reduce="mean")
    #print(rep_mean_2)
    #print(" ")
    return rep_mean

  def flip(self, msg, n_vars):
    return torch.cat([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], dim=0)

