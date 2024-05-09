import torch
from torch.utils.data import Dataset
from augmentation import Augmentation
import numpy as np
import os
import pickle
from solver import solve_sat
from augmentation import remove_space
from augmentation import get_num_literal 
def ilit_to_var_sign(x):
    #print(x)
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

# TODO(dhs): duplication
def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

def shift_ilit(x, offset):
    assert(x != 0)
    if x > 0: return x + offset
    else:     return x - offset


def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]


class SATInstances(Dataset):
    def __init__(self, args, start, data_length, labels, n_vars, augment = False):
        self.instance_folder = args.val_folder
        self.start_index = start
        self.len_data = data_length
        self.file_name = args.val_instance
        self.labels = labels
        self.n_vars = n_vars
        self.augment = augment
        self.data = args.data_source
        self.augment = args.supervised_augment
        self.at = args.add_trivial
        self.cr = args.clause_resolution
        self.ve = args.variable_elimination
        self.sub = args.subsume_clause
        self.be = args.blocked_clause
		
        self.gcl_cla = args.gcl_clause_drop
        self.gcl_var = args.gcl_var_drop
        self.gcl_link = args.gcl_link_purt
        self.gcl_sub = args.gcl_subgraph
        self.args = args
        """
        label_file = os.path.join(self.instance_folder, args.val_label)
        with open(label_file, 'rb') as handle:
            self.labels = pickle.load(handle)
        n_vars_file = os.path.join(self.instance_folder, args.val_n_vars)
        with open(n_vars_file, 'rb') as handle:
            self.n_vars = pickle.load(handle)
        """

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        file_name_index = None
        #if self.data == "sgen":
        #    file_name_index = os.path.join(self.instance_folder, self.file_name + "-" + str(index+self.start_index)+".pkl")
        #if self.data == "sr":
        file_name_index = os.path.join(self.instance_folder, self.file_name + str(index+self.start_index)+".pkl")
        with open(file_name_index, 'rb') as f:
            lines = pickle.load(f)
        nvar = self.n_vars[index]
        if self.augment:
            augment = Augmentation(instance=lines, at=self.at, cr=self.cr, ve=self.ve, be=self.be, sub=self.sub, 
								   gcl_cla=self.gcl_cla, gcl_var=self.gcl_var, gcl_link=self.gcl_link, gcl_sub=self.gcl_sub)
            lines, nvar = augment.augment_instance(self.args)
        return nvar, lines, self.labels[index]
        #return get_num_literal(lines), lines, self.labels[index]


class SATInstancesAugment(Dataset):
    def __init__(self, args, start, data_length, labels, n_vars, augment = False):
        self.instance_folder = args.val_folder
        self.start_index = start
        self.len_data = data_length
        self.file_name = args.val_instance
        self.labels = labels
        self.n_vars = n_vars
        self.augment = augment
        self.at = args.add_trivial
        self.cr = args.clause_resolution
        self.ve = args.variable_elimination
        self.sub = args.subsume_clause
        self.be = args.blocked_clause

        self.gcl_cla = args.gcl_clause_drop
        self.gcl_var = args.gcl_var_drop
        self.gcl_link = args.gcl_link_purt
        self.gcl_sub = args.gcl_subgraph

        self.args = args
        """
        label_file = os.path.join(self.instance_folder, args.val_label)
        with open(label_file, 'rb') as handle:
            self.labels = pickle.load(handle)
        n_vars_file = os.path.join(self.instance_folder, args.val_n_vars)
        with open(n_vars_file, 'rb') as handle:
            self.n_vars = pickle.load(handle)
        """

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        file_name_index = os.path.join(self.instance_folder, self.file_name + "_" + str(index+self.start_index)+".pkl")
        with open(file_name_index, 'rb') as f:
            lines = pickle.load(f)
        augment = Augmentation(instance=lines, at=self.at, cr=self.cr, ve=self.ve, be=self.be, sub=self.sub,
                               gcl_cla=self.gcl_cla, gcl_var=self.gcl_var, gcl_link=self.gcl_link, gcl_sub=self.gcl_sub)
        augment1, n_var1 = augment.augment_instance(self.args)
        return n_var1, augment1, self.n_vars[index], lines, self.labels[index]


class SATInstancesAugmentTwo(Dataset):

    def __init__(self, instances, args,  n_vars = None, is_sat=None, file_folder = None, num_file = None):
        self.instances = instances
        self.at = args.add_trivial
        self.cr = args.clause_resolution
        self.ve = args.variable_elimination
        self.be = args.blocked_clause
        self.sub = args.subsume_clause

        self.gcl_cla = args.gcl_clause_drop
        self.gcl_var = args.gcl_var_drop
        self.gcl_link = args.gcl_link_purt
        self.gcl_sub = args.gcl_subgraph

        self.ssl = args.ssl
        self.n_vars = n_vars
        self.is_sat = is_sat
        self.file_name = args.val_instance
        self.study_ve = args.study_ve
        self.file_folder = file_folder
        self.num_file = num_file
        self.args = args

    def __len__(self):
        if self.instances != None:
            return len(self.instances)
        else:
            return self.num_file

    def __getitem__(self, index):
        #print("here")
        if self.ssl:
            instance = None
            if self.instances != None:
                instance = self.instances[index]
            elif self.file_folder != None:
                file_name_index = os.path.join(self.file_folder, self.file_name + str(index) + ".pkl")
                with open(file_name_index, 'rb') as f:
                    instance = pickle.load(f)
            augment = Augmentation(instance=instance, at=self.at, cr=self.cr, ve=self.ve, be=self.be, sub=self.sub, 
									gcl_cla=self.gcl_cla, gcl_var=self.gcl_var, gcl_link=self.gcl_link, gcl_sub=self.gcl_sub)
            augment1, augment2, n_var1, n_var2 = augment.augment_two_instances(self.args)
            #print(len(instance), len(augment1), len(augment2))
            #if self.study_ve:
            return n_var1, augment1, n_var2, augment2, (len(augment1)+len(augment2) - 2*len(instance))/2
            #else:
            #    return n_var1, augment1, n_var2, augment2
        else:
            return self.n_vars[index], self.instances[index], self.is_sat[index]

class Problem(object):
    def __init__(self, n_vars, iclauses, n_cells_per_batch, offset, clause_offset, is_sat = None, added_clause = None):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)
        self.clause_offset = clause_offset
        self.is_sat = is_sat
        self.offset = offset
        self.n_cells = sum(n_cells_per_batch)
        self.n_cells_per_batch = n_cells_per_batch
        self.added_clause = added_clause
        self.compute_L_unpack(iclauses)

    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1
        assert(cell == self.n_cells)


def collate_fn_test(batch):
    all_iclauses = []
    all_is_sat = []
    all_n_cells = []
    clause_offset = []
    total_var = 0
    offset = []

    for n_vars, iclauses, is_sat in batch:
        offset.append(n_vars)
        all_iclauses.extend(shift_iclauses(iclauses, total_var))
        all_is_sat.append(is_sat)
        clause_offset.append(len(iclauses))
        all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
        total_var += n_vars

    return Problem(n_vars=total_var, iclauses=all_iclauses, is_sat=all_is_sat, n_cells_per_batch=all_n_cells, offset=offset, clause_offset=clause_offset)

def collate_fn_train_sup(batch):
    all_iclauses_1 = []
    all_n_cells_1 = []
    total_var_1 = 0
    offset_1 = []
    clause_offset_1 = []

    all_iclauses_2 = []
    all_n_cells_2 = []
    total_var_2 = 0
    offset_2 = []
    clause_offset_2 = []

    all_is_sat = []

    for n_vars_1, iclauses_1, n_vars_2, iclauses_2, is_sat in batch:
        offset_1.append(n_vars_1)
        all_iclauses_1.extend(shift_iclauses(iclauses_1, total_var_1))
        all_n_cells_1.append(sum([len(iclause) for iclause in iclauses_1]))
        total_var_1 += n_vars_1
        clause_offset_1.append(len(iclauses_1))

        offset_2.append(n_vars_2)
        all_iclauses_2.extend(shift_iclauses(iclauses_2, total_var_2))
        all_n_cells_2.append(sum([len(iclause) for iclause in iclauses_2]))
        total_var_2 += n_vars_2
        clause_offset_2.append(len(iclauses_2))

        all_is_sat.append(is_sat)

    prob1 = Problem(n_vars=total_var_1, iclauses=all_iclauses_1, n_cells_per_batch=all_n_cells_1, offset=offset_1, is_sat=all_is_sat, clause_offset=clause_offset_1)
    prob2 = Problem(n_vars=total_var_2, iclauses=all_iclauses_2, n_cells_per_batch=all_n_cells_2, offset=offset_2, is_sat=all_is_sat, clause_offset=clause_offset_2)

    return prob1, prob2


def collate_fn_train(batch):
    all_iclauses_1 = []
    all_n_cells_1 = []
    total_var_1 = 0
    offset_1 = []
    clause_offset_1 = []

    all_iclauses_2 = []
    all_n_cells_2 = []
    total_var_2 = 0
    offset_2 = []
    clause_offset_2 = []

    added_clause = 0

    for n_vars_1, iclauses_1, n_vars_2, iclauses_2, increase_clause in batch:
        added_clause += increase_clause
        offset_1.append(n_vars_1)
        all_iclauses_1.extend(shift_iclauses(iclauses_1, total_var_1))
        all_n_cells_1.append(sum([len(iclause) for iclause in iclauses_1]))
        total_var_1 += n_vars_1
        clause_offset_1.append(len(iclauses_1))

        offset_2.append(n_vars_2)
        all_iclauses_2.extend(shift_iclauses(iclauses_2, total_var_2))
        all_n_cells_2.append(sum([len(iclause) for iclause in iclauses_2]))
        total_var_2 += n_vars_2
        clause_offset_2.append(len(iclauses_2))

    prob1 = Problem(n_vars=total_var_1, iclauses=all_iclauses_1, n_cells_per_batch=all_n_cells_1, offset=offset_1, clause_offset = clause_offset_1, added_clause = added_clause)
    prob2 = Problem(n_vars=total_var_2, iclauses=all_iclauses_2, n_cells_per_batch=all_n_cells_2, offset=offset_2, clause_offset = clause_offset_2, added_clause = added_clause)

    return prob1, prob2

