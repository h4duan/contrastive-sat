import os
import sys
import argparse
import ipdb
import functools
import pickle
import numpy as np
import aiger as A
import aiger_cnf as ACNF
import aiger.common as cmn
import funcy as fn

from bidict import bidict
# from gen_utils import random_string
from cnf_tools import write_to_string, write_to_file

def aiger2cnf(circ, horizon, *, truth_strategy='all', fresh=None):
  if truth_strategy is None:
    truth_strategy = 'all'
  if fresh is None:
    max_var = 0

    def fresh(_):
      nonlocal max_var
      max_var += 1
      return max_var
  
  print('Using truth strategy: {}'.format(truth_strategy))
  step, old2new_lmap = circ.cutlatches()
  init = dict(old2new_lmap.values())
  states = set(init.keys())    
  state_inputs = [A.aig.Input(k) for k in init.keys()]
  clauses, seen_false, gate2lit = [], False, ACNF.cnf.SymbolTable(fresh)
  
  # Set up init clauses
  true_var = fresh(True)    
  clauses.append((true_var,))                
  tf_vars = {True: true_var, False: -true_var}
  for k,v in init.items():
    gate2lit[A.aig.Input(k)] = tf_vars[v]
      
  in2lit = bidict()
  outlits= []
  timestep_mapping = {}
  for time in range(horizon):
    # Only remember states.        
    gate2lit = ACNF.cnf.SymbolTable(fresh,fn.project(gate2lit, state_inputs))
    # start_len = len(gate2lit)
    for gate in cmn.eval_order(step.aig):
      if isinstance(gate, A.aig.Inverter):
        gate2lit[gate] = -gate2lit[gate.input]                
      elif isinstance(gate, A.aig.AndGate):
        clauses.append((-gate2lit[gate.left], -gate2lit[gate.right],  gate2lit[gate]))  # noqa
        clauses.append((gate2lit[gate.left],                         -gate2lit[gate]))  # noqa
        clauses.append((                       gate2lit[gate.right], -gate2lit[gate]))  # noqa
      elif isinstance(gate, A.aig.Input):
        if gate.name in states:      # We already have it from init or end of last round
          continue
        else:                 # This is a real output, add and remember it
          action_name = '{}_{}'.format(gate.name,time)
          in2lit[action_name] = gate2lit[gate]
    outlits.extend([gate2lit[step.aig.node_map[o]] for o in circ.aig.outputs])
    for s in states:
        assert step.aig.node_map[s] in gate2lit.keys()
        gate2lit[A.aig.Input(s)] = gate2lit[step.aig.node_map[s]]        
    for k,v in gate2lit.items():
      if abs(v) not in timestep_mapping.keys():
        timestep_mapping[abs(v)] = time
      # elif timestep_mapping[abs(v)]!=time:
      #   print('Not reassigning {} from {} to {}'.format(abs(v),timestep_mapping[abs(v)],time))
  if truth_strategy == 'all':
    for lit in outlits:
      clauses.append((lit,))
  elif truth_strategy == 'last':
    clauses.append((outlits[-1],))
  elif truth_strategy == 'any':
    clauses.append(tuple(outlits))
  else:
    raise "Help!"

  return ACNF.cnf.CNF(clauses, in2lit, outlits, None), timestep_mapping


def main(args):
  print('Parsing...')
  circuit = A.to_aig(args.file if args.file else sys.stdin.read())  
  print('\nUnrolling {} steps...\n'.format(args.timestep))
  cnf, step_mapping = aiger2cnf(circuit,args.timestep,truth_strategy=args.truth_strategy)
  maxvar = max([max([abs(y) for y in x]) for x in cnf.clauses])
  if args.out:
    write_to_file(maxvar, cnf.clauses,args.out)
    if args.annotate_step:
      annotation_fname = '{}.annt'.format(os.path.splitext(args.out)[0])
      with open(annotation_fname,'wb') as f:
        pickle.dump(step_mapping, f)
  else:
    print(write_to_string(maxvar, cnf.clauses))




if __name__=='__main__':
    sys.setrecursionlimit(5000)
    parser = argparse.ArgumentParser(description='Convert AIGER ascii format to CNF and annotate')    
    parser.add_argument('-d', '--destination_dir', type=str, default=os.curdir, help='destination directory')
    parser.add_argument('-s', '--truth_strategy', type=str, default=None, help='destination directory')
    parser.add_argument('-f', '--file', type=str, default=None, help='Read input from file')
    parser.add_argument('-o', '--out', type=str, default=None, help='Output to file')
    parser.add_argument('-t', '--timestep', type=int, default=1, help='Unroll horizon')
    parser.add_argument('--annotate_step', action='store_true', default=False, help='annotate timestep')
    args = parser.parse_args()
    main(args)

    
