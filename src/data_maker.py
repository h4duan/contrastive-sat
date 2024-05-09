import math
import os
import numpy as np
import random
import pickle
import sys
import argparse
import PyMiniSolvers.minisolvers as minisolvers
from random import randint
from solver import solve_sat
#from mk_problem import mk_batch_problem
from sat_dataset import SATInstances
from augmentation import remove_space
from augmentation import get_num_literal
#from sat_dataset import SATInstancesAugment
from sat_dataset import SATInstancesAugmentTwo
from sgen import sgen
from power_law import power_law
from double_power_law import double_power_law
from popularity_similarity import popularity_similarity
import os


def generate_k_iclause(n, k):
  vs = np.random.choice(n, size=min(n, k), replace=False)
  return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]


def gen_iclause_pair(args, n):
  solver = minisolvers.MinisatSolver()
  for i in range(n): solver.new_var(dvar=True)

  iclauses = []

  while True:
    k_base = 1 if random.random() < args.p_k_2 else 2
    k = k_base + np.random.geometric(args.p_geo)
    iclause = generate_k_iclause(n, k)

    solver.add_clause(iclause)
    is_sat = solver.solve()
    if is_sat:
      iclauses.append(iclause)
    else:
      break

  iclause_unsat = iclause
  iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
  return n, iclauses, iclause_unsat, iclause_sat


def generate_train(args, problem_len = None):

  if problem_len == None:
      problem_len = args.batch_size

  instances = []
  labels = []
  n_vars = []
  n_var_max = args.max_n
  n_var_min = args.min_n

  for n_var in range(n_var_min, n_var_max + 1, args.nvar_step):
    for _ in range(problem_len):
      var, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(args, n_var)
      #n_vars += [var]

      if random.random() < 0.5:
        iclauses.append(iclause_unsat)
        labels += [0]
      else:
        iclauses.append(iclause_sat)
        labels +=[1]

      instance = remove_space(iclauses)
      n_vars += [get_num_literal(instance)]
      instances.append(instance)

  return SATInstancesAugmentTwo(instances=instances, args = args, n_vars = n_vars, is_sat = labels)

def generate_train_sgen(args, problem_len=None):
  if problem_len == None:
      problem_len = args.batch_size

  instances = []
  labels = None
  n_vars = []
  n_var_max = args.max_n
  n_var_min = args.min_n

  for n_var in range(n_var_min, n_var_max + 1, args.nvar_step):
      for _ in range(problem_len):
        i = randint(0, 10000)
        j = randint(0, 10000)
        h = int(str(i) + str(j))
        sgen(n_var, ".", "UNSAT", h)
        file_name = "sgen-unsat-" + str(n_var) + "-" + str(h) + ".cnf"
        #print(file_name)
        instance = open(file_name)
        lines = instance.readlines()
        clauses = []
        for line in lines:
          if line[0] != 'c' and line[0] != 'p':
            if random.uniform(0, 1) < 0.9965:
              line = line.strip().split()
              line = line[:-1]
              #print(line)
              line = [int(x) for x  in line]
              clauses.append(line)
            else:
              #print("change")
              clause = line.strip().split()
              # clause = [int(x) for x in clause]
              clause = np.random.choice(n_var+1, 3) + 1
              if random.uniform(0, 1) < 0.5:
                clause = -1 * clause
              clauses.append(clause.tolist())
        os.remove(file_name)
        if len(clauses) == 0:
            print(lines)
            print(file_name)
        instance = remove_space(clauses)
        n_vars += [get_num_literal(instance)]
        instances.append(instance)

  return SATInstancesAugmentTwo(instances=instances, args=args, n_vars=n_vars, is_sat=labels)

def generate_train_power_law(args, problem_len=None):
  if problem_len == None:
      problem_len = args.batch_size

  instances = []
  labels = None
  n_vars = []
  n_var_max = args.max_n
  n_var_min = args.min_n

  for n_var in range(n_var_min, n_var_max + 1, args.nvar_step):
      for _ in range(problem_len):
        i = randint(0, 10000)
        j = randint(0, 10000)
        h = int(str(i) + str(j))
        #power_law(n_var, ".", "UNSAT", h)
        file_name = "power_law" + "-" + str(h)
        power_law(file_name, h)
        #print(file_name)
        instance = open(file_name+".cnf")
        lines = instance.readlines()
        clauses = []
        for line in lines:
          if line[0] != 'c' and line[0] != 'p':
              line = line.strip().split()
              line = line[:-1]
              #print(line)
              line = [int(x) for x  in line]
              clauses.append(line)
        os.remove(file_name+".cnf")
        instance = remove_space(clauses)
        n_vars += [get_num_literal(instance)]
        instances.append(instance)

  return SATInstancesAugmentTwo(instances=instances, args=args, n_vars=n_vars, is_sat=labels)

def generate_train_double_power_law(args, problem_len=None):
  if problem_len == None:
      problem_len = args.batch_size

  instances = []
  labels = None
  n_vars = []
  n_var_max = args.max_n
  n_var_min = args.min_n

  for n_var in range(n_var_min, n_var_max + 1, args.nvar_step):
      for _ in range(problem_len):
        i = randint(0, 10000)
        j = randint(0, 10000)
        h = int(str(i) + str(j))
        #power_law(n_var, ".", "UNSAT", h)
        file_name = "double_power_law" + "-" + str(h)
        double_power_law(file_name, h)
        #print(file_name)
        instance = open(file_name+".cnf")
        lines = instance.readlines()
        clauses = []
        for line in lines:
          if line[0] != 'c' and line[0] != 'p':
              line = line.strip().split()
              if len(line) == 1:
                  continue 
              line = line[:-1]
              #print(line)
              line = [int(x) for x  in line]
              clauses.append(line)
        os.remove(file_name+".cnf")
        instance = remove_space(clauses)
        n_vars += [get_num_literal(instance)]
        instances.append(instance)

  return SATInstancesAugmentTwo(instances=instances, args=args, n_vars=n_vars, is_sat=labels)

def generate_train_popularity_similarity(args, problem_len=None):
  if problem_len == None:
      problem_len = args.batch_size

  instances = []
  labels = None
  n_vars = []
  n_var_max = args.max_n
  n_var_min = args.min_n

  for n_var in range(n_var_min, n_var_max + 1, args.nvar_step):
      for _ in range(problem_len):
        i = randint(0, 10000)
        j = randint(0, 10000)
        h = int(str(i) + str(j))
        #power_law(n_var, ".", "UNSAT", h)
        file_name = "popularity" + "-" + str(h) + ".cnf"
        popularity_similarity(file_name, h)
        #print(file_name)
        instance = open(file_name)
        lines = instance.readlines()
        clauses = []
        for line in lines:
          if line[0] != 'c' and line[0] != 'p':
              line = line.strip().split()
              if len(line) == 1:
                  continue
              line = line[:-1]
              #print(line)
              line = [int(x) for x  in line]
              clauses.append(line)
        os.remove(file_name)
        instance = remove_space(clauses)
        n_vars += [get_num_literal(instance)]
        instances.append(instance)

  return SATInstancesAugmentTwo(instances=instances, args=args, n_vars=n_vars, is_sat=labels)

def generate_test(args):
  instances = []
  labels = []
  n_vars = []
  n_var_max = args.max_n
  n_var_min = args.min_n
  problem_len = int(args.num_data / ((args.max_n - args.min_n) / args.nvar_step))

  for n_var in range(n_var_min, n_var_max+1, args.nvar_step):
    for _ in range(problem_len):
      var, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(args, n_var)
      n_vars += [var]

      if random.random() < 0.5:
        iclauses.append(iclause_unsat)
        labels += [0]
      else:
        iclauses.append(iclause_sat)
        labels += [1]

      instances.append(iclauses)

  return SATInstances(instances=instances, labels=labels, n_vars = n_vars)


