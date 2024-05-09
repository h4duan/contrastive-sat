import os
import random
import time
import noise
import functools
import pickle
import numpy as np
import aiger as A
import aiger_bv as BV
import aiger_coins as C
import aiger_gridworld as GW
import aiger_ptltl as LTL
import aiger_cnf as ACNF
import aiger.common as cmn
import funcy as fn
import matplotlib.pyplot as plt

from gen_types import FileName
from bidict import bidict
from samplers.sampler_base import SamplerBase
from random import randint, seed
from gen_utils import random_string


COLOR_ALIAS = {
    'yellow': '#ffff8c', 'brown': '#ffb081',
    'red': '#ff5454', 'blue': '#9595ff'
}

def tile(color='black'):
    color = COLOR_ALIAS.get(color, color)
    s = '&nbsp;'*4
    return f"<text style='border: solid 1px;background-color:{color}'>{s}</text>"

def ap_at_state(x, y, sensor, in_ascii=False):
    """Use sensor to create colored tile."""
    state = encode_state(x, y)
    obs = sensor(state)[0]   # <----------   

    for k in COLOR_ALIAS.keys():
        if obs[k][0]:
            return tile(k)
    return tile('white')

def get_mask_test(X, Y):
  def f(xmask, ymask):
    return ((X & xmask) !=0) & ((Y & ymask) != 0)
  return f


def create_sensor(aps):
  sensor = BV.aig2aigbv(A.empty())
  for name, ap in aps.items():
      sensor |= ap.with_output(name).aigbv
  return sensor

def get_noise_grid(base=0, shape=(16,16), scale=100., octaves=6, persistence=0.5, lacunarity=2.0):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.snoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=base)
    return world

def world_from_noise(noise_grid):
    def map_perlin_to_colors(val):
        if val < -0.27:
            return 3          # 'red'
        elif val < 0.15:
            return 0          # white,
        elif val < 0.2:
            return 1        # yellow
        elif val < 0.38:
            return 4          # blue
        else: 
            return 2
    
    
    world = np.zeros(noise_grid.shape)
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            world[i][j] = map_perlin_to_colors(noise_grid[i][j])    
    return world    

def spec2monitor(spec):
  monitor = spec.aig | A.sink(['red', 'yellow', 'brown', 'blue'])
  monitor = BV.aig2aigbv(monitor)
  return monitor

def mdp2cnf(circ, horizon, *, fresh=None, truth_strategy='last'):
    if fresh is None:
        max_var = 0

        def fresh(_):
            nonlocal max_var
            max_var += 1
            return max_var
    
    
    imap = circ.imap
    inputs = circ.inputs
    step, old2new_lmap = circ.cutlatches()
    init = dict(old2new_lmap.values())
    init = step.imap.blast(init)
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
        start_len = len(gate2lit)
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
        for v in gate2lit.values():          
          timestep_mapping[abs(v)] = time
    if truth_strategy == 'all':
        for lit in outlits:
            clauses.append((lit,))
    elif truth_strategy == 'last':
        clauses.append((outlits[-1],))
    else:
        raise "Help!"

    return ACNF.cnf.CNF(clauses, in2lit, outlits, None), timestep_mapping

class GridSampler(SamplerBase):
  def __init__(self, size=8, horizon_min=2, horizon_max=2, annotate=False, **kwargs):
    SamplerBase.__init__(self, **kwargs)
    self.size = int(size)
    self.horizon_min = int(horizon_min)
    self.horizon_max = int(horizon_max)
    self.annotate = annotate
    self.X = BV.atom(self.size, 'x', signed=False)
    self.Y = BV.atom(self.size, 'y', signed=False)
    self.mask_test = get_mask_test(self.X, self.Y)

  def encode_state(self, x, y):
    x, y = [BV.encode_int(self.size, 1 << (v - 1), signed=False) for v in (x, y)]
    return {'x': tuple(x), 'y': tuple(y)}


  def make_spec(self):
    LAVA, RECHARGE, WATER, DRY = map(LTL.atom, ['red', 'yellow', 'blue', 'brown'])

    EVENTUALLY_RECHARGE = RECHARGE.once()
    AVOID_LAVA = (~LAVA).historically()

    RECHARGED_AND_ONCE_WET = RECHARGE & WATER.once()
    DRIED_OFF = (~WATER).since(DRY)

    DIDNT_RECHARGE_WHILE_WET = (RECHARGED_AND_ONCE_WET).implies(DRIED_OFF)
    DONT_RECHARGE_WHILE_WET = DIDNT_RECHARGE_WHILE_WET.historically()

    CONST_TRUE = LTL.atom(True)


    SPECS = [
      CONST_TRUE, AVOID_LAVA, EVENTUALLY_RECHARGE, DONT_RECHARGE_WHILE_WET,
      AVOID_LAVA & EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET,
      AVOID_LAVA & EVENTUALLY_RECHARGE,
      AVOID_LAVA & DONT_RECHARGE_WHILE_WET,
      EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET,
    ]

    SPEC_NAMES = [
      "CONST_TRUE", "AVOID_LAVA", "EVENTUALLY_RECHARGE", "DONT_RECHARGE_WHILE_WET",
      "AVOID_LAVA & EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET",
      "AVOID_LAVA & EVENTUALLY_RECHARGE",
      "AVOID_LAVA & DONT_RECHARGE_WHILE_WET",
      "EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET",
    ]

    return AVOID_LAVA
    # return {k: v for (v,k) in zip(SPECS,SPEC_NAMES)}

  def get_feature_mask(self, feat, world, pairs_only=False):
    size = world.shape[1]
    mask_func = self.mask_test
    feat_rows = [feat in x for x in world]
    if np.sum([feat in x for x in world.transpose()]) < np.sum(feat_rows):
      feat_mask_tuples= self.get_feature_mask(feat,world.transpose(), pairs_only=True)
      flip = lambda f: lambda *a: f(*reversed(a))
      mask_func = flip(mask_func)
    else:
      def to_int(row, size):
        return int(sum([2**i for (x,i) in zip(reversed(row),np.arange(size)) if x]))
        
      feat_mask_tuples = []
      for (x,row) in zip(np.where(feat_rows)[0]+1, (world[feat_rows]==feat)):
        row_mask = (int(1 << x-1), to_int(row, size))
        feat_mask_tuples.append(row_mask)
    if pairs_only:
      return feat_mask_tuples
    if not len(feat_mask_tuples):
      return mask_func(0,0)
    else:   # Return the actual circuit for the mask
      return functools.reduce(lambda x,y: x | y,fn.map(lambda tup: mask_func(*tup), feat_mask_tuples))


  def get_random_masks(self, world):
    random_aps = {
      'yellow': self.get_feature_mask(1, world), 'red': self.get_feature_mask(3,world), 
      'blue': self.get_feature_mask(4,world), 'brown': self.get_feature_mask(2,world)
    }
    
    return random_aps
  
  def make_grid(self, seed=None, cutoff=False):
    bignum = 2**16
    if not seed:
      seed = (int(time.time()) % bignum) * (os.getpid() % bignum)
    random.seed(seed)
    base = random.randint(1,2**16)
    world = world_from_noise(get_noise_grid(shape=(self.size, self.size),base=base, persistence=5.0, lacunarity=2.0, scale=100))
    x = random.randint(1,self.size)
    y = random.randint(1,self.size)
    DYN = GW.gridworld(self.size, start=(x, y), compressed_inputs=True)
    APS = self.get_random_masks(world)
    # APS = {       #            x-axis       y-axis
    #   'yellow': self.mask_test(0b1000_0001, 0b1000_0001),
    #   'blue':   self.mask_test(0b0001_1000, 0b0011100),
    #   'brown':   self.mask_test(0b0011_1100, 0b1000_0001),
    #   'red':    self.mask_test(0b1000_0001, 0b0100_1100) \
    #           | self.mask_test(0b0100_0010, 0b1100_1100),
    # }

    SENSOR = create_sensor(APS)
    if cutoff:
      return SENSOR
    spec = self.make_spec()
    MONITOR = spec2monitor(spec)
    circuit = DYN >> SENSOR >> MONITOR
    horizon = random.randint(self.horizon_min,self.horizon_max)
    # if self.annotate:

    #   def timed_fresh(_):
    #     nonlocal max_var
    #     max_var += 1
    #     return max_var

    cnf, step_mapping = mdp2cnf(circuit,horizon)
    return cnf, step_mapping, world, (x,y), horizon, seed
    
  def sample(self, stats_dict: dict) -> (FileName, FileName):
    fcnf, step_mapping, world, start_pos, eff_horizon, seed = self.make_grid()
    name = 'grid_{}_{}_{}_{}'.format(self.size,eff_horizon,random_string(8),seed+os.getpid())
    fname = '/tmp/{}.cnf'.format(name)
    self.write_expression(fcnf, fname, is_cnf=True)
    if self.annotate:
      annotation_fname = '/tmp/{}.annt'.format(name)
      with open(annotation_fname,'wb') as f:
        pickle.dump((world, start_pos, step_mapping), f)
    else:
      annotation_fname = None
    return fname, annotation_fname

