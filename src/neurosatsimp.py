import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch_scatter import scatter

from mlp import MLP


class NeuroSATSimp(nn.Module):
    def __init__(self, args):
        super(NeuroSATSimp, self).__init__()
        self.args = args
        self.bipartite = args.bipartite
        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False
        self.L_project = [nn.Linear(self.args.dim, self.args.dim)]
        self.C_project = None

        self.instance_norm_weight_1 = nn.Parameter(torch.ones(self.args.dim))
        self.instance_norm_bias_1 = nn.Parameter(torch.zeros(self.args.dim))
        self.instance_norm_weight_2 = nn.Parameter(torch.ones(self.args.dim))
        self.instance_norm_bias_2 = nn.Parameter(torch.zeros(self.args.dim))

        self.L_init = nn.Linear(1, args.dim)
        self.C_init = None
        if self.bipartite:
            self.C_init = nn.Linear(1, args.dim)

        self.L_msg = [MLP(self.args.dim, self.args.dim, self.args.dim)]
        self.C_msg = None
        if self.bipartite:
            self.C_msg = [MLP(self.args.dim, self.args.dim, self.args.dim)]
            self.C_project = [MLP(self.args.dim, self.args.dim, self.args.dim)]

        for _ in range(self.args.n_rounds - 1):
            self.L_msg.append(MLP(self.args.dim, self.args.dim, self.args.dim))
            self.L_project.append(nn.Linear(self.args.dim, self.args.dim))
            if args.bipartite:
                self.C_msg.append(MLP(self.args.dim, self.args.dim, self.args.dim))
                self.C_project.append(MLP(self.args.dim, self.args.dim, self.args.dim))

        self.L_msg = nn.ModuleList(self.L_msg)
        self.L_project = nn.ModuleList(self.L_project)
        if self.bipartite:
            self.C_msg = nn.ModuleList(self.C_msg)
            self.C_project = nn.ModuleList(self.C_project)
        # variance = scatter(torch.square(C_state), clause_index, dim=0, reduce="mean")  - torch.square(mean)
        """ 
    	self.L_layernorm = LayerNorm(self.args.dim)
    	self.C_layernorm = None
    	if args.bipartite:
        	self.C_layernorm = LayerNorm(self.args.dim)
    	"""

        self.L_vote = MLP(self.args.dim, self.args.dim, self.args.dim)

        self.denom = torch.sqrt(torch.Tensor([self.args.dim]))

    def forward(self, problem):
        device = self.device_param.device

        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.offset)
        n_variable_offset = problem.offset
        n_clause_offset = problem.clause_offset
        # print(n_vars, n_lits, n_clauses, n_probs)
        # literal_index = torch.zeros((n_vars, 1)).to(device).long()
        # clause_index = torch.zeros((n_clauses, 1)).to(device).long()
        literal_index = torch.repeat_interleave(torch.arange(n_probs).to(device),
                                                torch.tensor(n_variable_offset).to(device))
        literal_index = torch.reshape(literal_index, (n_vars, 1))
        literal_index = torch.cat((literal_index, literal_index), 0)

        clause_index = torch.repeat_interleave(torch.arange(n_probs).to(device),
                                               torch.tensor(n_clause_offset).to(device))
        clause_index = torch.reshape(clause_index, (n_clauses, 1))

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()

        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells), torch.Size([n_lits, n_clauses])).to(device)

        degree_literal = torch.sparse.mm(L_unpack, torch.reshape(torch.ones(n_clauses).to(device), (n_clauses, 1)))
        degree_clause = torch.sparse.mm(L_unpack.t(), torch.reshape(torch.ones(n_lits).to(device), (n_lits, 1)))
        # degree_literal = (degree_literal - torch.mean(degree_literal)) / (torch.std(degree_literal) + 1)
        # degree_clause = (degree_clause - torch.mean(degree_clause)) / (torch.std(degree_clause) + 1)
        L_state = self.L_init(degree_literal)
        C_state = None
        if self.bipartite:
            C_state = self.C_init(degree_clause)
        else:
            C_state = self.L_init(degree_clause)

        # L_state = self.L_init(torch.ones((n_lits, 1)).to(device))
        # C_state = self.L_init(torch.ones((n_clauses, 1)).to(device))
        for i in range(self.args.n_rounds):
            # L_pre_msg = self.L_msg[i](L_state)
            LC_msg = torch.sparse.mm(L_unpack.t(), self.L_project[i](L_state))
            old_C_state = torch.clone(C_state)
            C_state = LC_msg
            # C_state = self.L_layernorm(C_state)

            mean = scatter(C_state, clause_index, dim=0, reduce="mean")
            clause_offset = torch.tensor(n_clause_offset).to(device)
            mean = torch.repeat_interleave(mean, clause_offset, dim=0)
            epsilon = 1e-6
            sub = C_state - mean
            variance = scatter(torch.square(sub), clause_index, dim=0, reduce="mean")

            std = torch.sqrt(variance + epsilon)
            std = torch.repeat_interleave(std, clause_offset, dim=0)
            C_state = self.instance_norm_weight_1 * ((C_state - mean) / std) + self.instance_norm_bias_1
            if self.bipartite:
                C_state = self.C_msg[i](C_state) + old_C_state
            else:
                C_state = self.L_msg[i](C_state) + old_C_state

        CL_msg = None
        if self.bipartite:
            CL_msg = torch.sparse.mm(L_unpack, self.C_project[i](C_state))
        else:
            CL_msg = torch.sparse.mm(L_unpack, self.L_project[i](C_state))
        old_L_state = torch.clone(L_state)
        L_state = CL_msg + self.flip(L_state, n_vars)
        #L_state = self.L_layernorm(L_state)
        mean = scatter(L_state, literal_index, dim=0, reduce="mean")
        literal_offset = torch.tensor(n_variable_offset).cuda()
        mean = torch.repeat_interleave(mean, literal_offset, dim = 0)
        mean = torch.cat((mean, mean), 0)
        sub = L_state - mean
        variance = scatter(torch.square(sub), literal_index, dim = 0, reduce="mean")
        #variance = scatter(torch.square(L_state), literal_index, dim=0, reduce="mean")  - torch.square(mean)
        epsilon = 1e-6
        #print(mean)
        #print(torch.count(variance[variance < 0]))
        #print(" ")
        std = torch.sqrt(variance + epsilon)
        #print(torch.sum(torch.isnan(std)))
        #pos_literal_index = literal_index[:n_vars, :]
        literal_offset = torch.tensor(n_variable_offset).cuda()
        #mean = torch.repeat_interleave(mean, literal_offset, dim = 0)
        std = torch.repeat_interleave(std, literal_offset, dim = 0)
        #mean = torch.cat((mean, mean), 0)
        std = torch.cat((std, std), 0)
        L_state = self.instance_norm_weight_2 * ((L_state - mean) / std) + self.instance_norm_bias_2
        L_state = self.L_msg[i](L_state) + old_L_state
        #print(L_state.shape)
        #print(" ")
        logits = L_state.squeeze(0)
        rep_mean = scatter(logits, literal_index, dim=0, reduce="mean")
        #print(rep_mean)
        #print(" ")
        return self.L_vote(rep_mean)

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)
