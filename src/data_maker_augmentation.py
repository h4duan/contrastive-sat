import math
import os
import numpy as np
import random
import pickle
import sys
import argparse
import PyMiniSolvers.minisolvers as minisolvers
from solver import solve_sat
from mk_problem import mk_batch_problem
from augmentation import Augmentation

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

def generate_augmentations(args):
  #f = open(args.gen_log, 'w')

  n_cnt = args.max_n - args.min_n + 1
  problems_per_n = args.n_pairs * 1.0 / n_cnt

  problems_1 = []
  batches_1 = []
  n_nodes_in_batch_1 = 0
  prev_n_vars_1 = None

  problems_2 = []
  batches_2 = []
  n_nodes_in_batch_2 = 0
  prev_n_vars_2 = None


  for n_var in range(args.min_n, args.max_n+1):
    lower_bound = int((n_var - args.min_n) * problems_per_n)
    upper_bound = int((n_var - args.min_n + 1) * problems_per_n)
    for problems_idx in range(lower_bound, upper_bound):
      n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(args, n_var)

      if random.random() < 0.5:
        iclauses.append(iclause_unsat)
      else:
        iclauses.append(iclause_sat)

      augment1, augment2, n_vars_1, n_vars_2 = Augmentation(iclauses, at = args.add_trivial, ve = args.variable_elimination, be = args.blocked_clause, cr = args.clause_resolution,
                                                            gcl_cla=args.gcl_cla, gcl_var=args.gcl_var, gcl_link=args.gcl_link, gcl_sub=args.gcl_sub).augment_two_instances()
      #print(augment1)
      n_clauses = len(augment1)
      n_cells = sum([len(iclause) for iclause in augment1])
      n_nodes_1 = 2 * n_vars + n_clauses

      n_clauses = len(augment1)
      n_cells = sum([len(iclause) for iclause in augment2])
      n_nodes_2 = 2 * n_vars + n_clauses

      n_nodes = max(n_nodes_1, n_nodes_2)

      if n_nodes > args.max_nodes_per_batch:
        continue

      batch_ready = False
      if (args.one and len(problems_1) > 0):
       batch_ready = True
      #elif (prev_n_vars and n_vars != prev_n_vars):
      #  batch_ready = True
      elif (not args.one) and (n_nodes_in_batch_1 + n_nodes_1 > args.max_nodes_per_batch or n_nodes_in_batch_2 + n_nodes_2 > args.max_nodes_per_batch):
        batch_ready = True

      if batch_ready:
        batches_1.append(mk_batch_problem(problems_1))
        batches_2.append(mk_batch_problem(problems_2))
        #print("batch %d done (%d vars, %d problems)..." % (len(batches), prev_n_vars, len(problems)))
        del problems_1[:]
        del problems_2[:]
        n_nodes_in_batch_1 = 0
        n_nodes_in_batch_2 = 0

      #prev_n_vars = n_vars

      is_sat_1, stats_1 = solve_sat(n_vars_1, augment1)
      is_sat_2, stats_2 = solve_sat(n_vars_2, augment2)
      assert(is_sat_1 == is_sat_2)
      #print(augment1)
      problems_1.append(("sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d_sat=0" % (n_vars_1, args.p_k_2, args.p_geo, problems_idx), n_vars_1, augment1, is_sat_1))
      problems_2.append(("sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d_sat=0" % (n_vars_2, args.p_k_2, args.p_geo, problems_idx), n_vars_2, augment2, is_sat_2))
      n_nodes_in_batch_1 += n_nodes_1
      n_nodes_in_batch_2 += n_nodes_2 

  if len(problems_1) > 0:
    batches_1.append(mk_batch_problem(problems_1))
    batches_2.append(mk_batch_problem(problems_2))
    #print("batch %d done (%d vars, %d problems)..." % (len(batches), n_vars, len(problems)))
    del problems_1[:]
    del problems_2[:]
  
  return batches_1, batches_2


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('out_dir', action='store', type=str)
  parser.add_argument('gen_log', action='store', type=str)
  parser.add_argument('n_pairs', action='store', type=int)
  parser.add_argument('max_nodes_per_batch', action='store', type=int)

  parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
  parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)

  parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
  parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)

  parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
  parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)

  parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

  parser.add_argument('--one', action='store', dest='one', type=int, default=0)
  parser.add_argument('--max_dimacs', action='store', dest='max_dimacs', type=int, default=None)

  args = parser.parse_args()

  if args.py_seed is not None: random.seed(args.py_seed)
  if args.np_seed is not None: np.random.seed(args.np_seed)
  
  batch1, batch2 = generate_augmentations(args)

  # create directory
  # if not os.path.exists(args.out_dir):
  #   os.mkdir(args.out_dir)

  dataset_filename = args.out_dir
  #print("Writing %d batches to %s..." % (len(batches), dataset_filename))
  with open(dataset_filename, 'wb') as f_dump:
    pickle.dump(batches, f_dump)
