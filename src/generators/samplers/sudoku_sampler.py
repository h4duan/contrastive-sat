"""
SUDOKU (NUMBER PLACE) PUZZLE GENERATOR
by Arel Cordero November 12, 2005

This program is released into the public domain.
Revision 3
"""
import os
import time
import pickle
import random, copy
import numpy as np

from gen_types import FileName
from samplers.sampler_base import SamplerBase
from gen_utils import random_string

from pysat.solvers import Glucose3
from pysat.formula import CNF
import sys
import itertools
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
log = logging.getLogger(__name__)

def negate(literal):
    return literal*(-1)

def rcn2var(size,row,col,num):
    """Return the variable which represents a (row,col,num)."""
    var = row * size**2 + col * size + num + 1 # +1 to avoid 0
    return var

def var2rcn(size, var):
    var = var.abs() - 1
    res = var % size**2
    row = (var - res) / size**2
    num = res % size
    col = (res - num) / size
    return row, col, num

def remove_space(instance):
    sorted_literals = np.unique(np.absolute(np.asarray([item for sublist in instance for item in sublist])))
    num_literals = len(sorted_literals)

    if sorted_literals[-1] == num_literals:
        return instance

    wrong_index = np.argwhere(sorted_literals != np.arange(1, num_literals + 1))[0]
    num_wrong_literal = sorted_literals[wrong_index]
    for i in range(len(instance)):
        for j in range(len(instance[i])):
            if abs(instance[i][j]) < num_wrong_literal:
                continue
            else:
                correct_id = list(sorted_literals).index(abs(instance[i][j])) + 1
                if instance[i][j] > 0:
                    instance[i][j] = correct_id
                else:
                    instance[i][j] = -correct_id

    return instance

def unit_propagation(instance):
    def has_unit(instance):
        for i in range(len(instance)):
            if len(instance[i]) == 1:
                # print(instance[i])
                return True
        return False

    while has_unit(instance):
        for c in range(len(instance)):
            if len(instance[c]) != 1:
                continue
            v = instance[c][0]
            delete_clause = []
            for j in range(len(instance)):
                if v in instance[j]:
                    delete_clause += [j]
                if -v in instance[j]:
                    instance[j].remove(-v)
            instance = [i for j, i in enumerate(instance) if j not in delete_clause]
            break

    for c_i, c in reversed(list(enumerate(instance))):
        if len(c) == 0:
            del instance [c_i]

    return instance

class CNF_formula:
    def __init__(self,encoding_nvars):
        self.clauses = []
        self.nvars = encoding_nvars

    def print_dimacs(self, fname = None):
        """Write the cnf formula with dimacs format
        in the fname file.
        """
        if fname == None:
            import sys
            f  = sys.stdout
        else:
            f = open(fname,'w')

        def print_dimacs_clause(clause):
            f.write(" ".join(map(str,clause))+" 0\n")

        up_cla = unit_propagation(self.clauses)
        if len(up_cla) != 0:
            up_cla = remove_space(up_cla)
            n_cla_up = len(up_cla)
            n_var_up = max([max([abs(l) for l in c]) for c in up_cla])

            f.write("c\nc CNF Sudoku Translation\nc\np cnf " + str(n_var_up)+" "
                    + str(n_cla_up) + "\n")
            for clause in up_cla:
                print_dimacs_clause(clause)

        if fname is not None:
            f.close()

    def num_vars(self):
        return self.nvars

    def get_clauses(self):
        return self.clauses

    def implies(self,a,b):
        self.clauses.append([a*(-1),b])

    def biimplies(self,a,b):
        self.implies(a,b)
        self.implies(b,a)

    def at_least_one(self,B):
        self.clauses.append(B)

    def at_most_one(self,B):
        for combination in itertools.combinations(range(len(B)),2):
            self.clauses.append([(-1)*B[combination[0]],(-1)*B[combination[1]]])

    def addclause(self,cl):
        self.clauses.append(cl)

    def exacly_one(self,B, classic=True):
        """ Exacly one restriction.
        """

        if classic:
            self.at_least_one(B)
            self.at_most_one(B)

        else:
            n = len(B)
            R = list(range(self.nvars,self.nvars + n))
            R[0]=None
            self.nvars += (n - 1)
            #Rn -> Rn-1
            for i in range(2,n):self.implies(R[i],R[i-1])
            #B1 <-> no(R2)
            self.biimplies(B[0],negate(R[1]))
            #Bn <-> Rn
            self.biimplies(B[-1],R[-1])
            #Bi <-> Ri and no(Ri+1)
            for i in range(1,n - 1):
                self.clauses.extend([[negate(B[i]),R[i]],
                            [negate(B[i]), negate(R[i+1])],
                            [negate(R[i]),R[i+1],B[i]]])

class SudokuCNF:
    def __init__(self, order, puzzle, classic=True):
        self.order = order
        self.size  = order**2
        self.sudoku = puzzle
        self.classic = classic
        self.var2rcn = {}

    def __eq__(self, other):
        if (other == None): return False
        if (self.order != other.order): return False
        for (i, j) in itertools.product(range(self.size), range(self.size)):
            if (self.sudoku[i, j] != other.sudoku[i, j]): return False

        return True

    def variable(self,row,col,num):
        """Return the variable which represents a (row,col,num)."""
        var = row * self.size**2 + col * self.size + num + 1 # +1 to avoid 0
        self.var2rcn[var] = (row, col, num)
        return var

    def as_features(self, encoding='binary'):
        assert encoding=='binary'
        puz = self.sudoku.copy()
        puz[puz==-1]=self.size
        return np.stack([puz==i for i in range(self.size+1)], axis=2)

    def save_annotation(self, fname):
        with open(fname, 'wb+') as f:
            pickle.dump(self.sudoku,f)

    def extended_translate_to_CNF(self):
        """ Translate to CNF all restrictions with all variales"""

        cnff = CNF_formula(self.size**3)
        for i in range(self.size):
            for j in range(self.size):
                A = [  self.variable(i,j,k) for k in range(self.size)]
                cnff.exacly_one(A, self.classic)# Cell restrictions
                B = [  self.variable(i,k,j) for k in range(self.size)]
                cnff.exacly_one(B, self.classic)# Row restrictions
                C = [  self.variable(k,i,j) for k in range(self.size)]
                cnff.exacly_one(C, self.classic)# Col restrictions
                if self.sudoku[i][j] != -1: #Fixed variables restrictions
                    cnff.addclause([self.variable(i,j,int(self.sudoku[i][j]))])

        #Region restrictions
        for (block_i, block_j) in itertools.product(range(self.order), range(self.order)):
            i, j = block_i * self.order, block_j * self.order
            for num in range(self.size):
                D = [ self.variable(i+i_inc,j+j_inc,num)  for i_inc in range(self.order)
                      for j_inc in range(self.order) ]
                cnff.exacly_one(D, self.classic)

        return cnff,[i for i in range((self.size**3)+1)]

    def display(self):
        for row in self.sudoku:
            print(' '.join([f'{n+1:>2}' if n+1 else '--' for n in row]))

    def solve(self):
        cnf = self.encode()
        glucose = Glucose3()
        glucose.append_formula(cnf.get_clauses())
        glucose.solve()
        model = glucose.get_model()

        if (model): return self.fit(model)

        return None

    def encode(self):
        cnf, decoding_map = self.extended_translate_to_CNF()

        return cnf

    def fit_partial(self, partial_model):
        model = np.full(self.size**3, -1)
        for lit in partial_model:
            model[abs(lit) - 1] = lit

        return self.fit(model)

    def fit(self, model):
        self.encode()

        decoded = np.full((self.size, self.size), -1, dtype=int)
        for var in range(self.size**3):
            if (model[var] < 0): continue # The variable is False

            row, col, num = self.var2rcn[model[var]]
            assert(decoded[row][col] == -1), f"ERROR: Reassigning puzzle[{row}][{col}] = {decoded[row][col]} to {num}"
            decoded[row][col] = num

        return SudokuCNF(self.order, decoded)

    def pluck(self, count):
        ind = np.hstack(([0] * count, [1] * (self.size**2 - count)))
        np.random.shuffle(ind)
        mask = ind.reshape((self.size, self.size))
        puzzle = mask * (self.sudoku + 1) - 1
        return SudokuCNF(self.order, puzzle)

    def random_swap(self, swaps):
        puzzle = self.sudoku.copy()
        ind = np.nonzero(puzzle + 1)
        ind = list(zip(ind[0], ind[1]))
        for _ in range(swaps):
            c_i, c_j = np.random.choice(len(ind), 2)
            i = ind[c_i]
            j = ind[c_j]

            tmp = puzzle[i]
            puzzle[i] = puzzle[j]
            puzzle[j] = tmp

        return SudokuCNF(self.order, puzzle)

    # def pluck(self, count): # This implementatoin makes sure to only pluck the non-empty cells
    #     full_idx = np.where(self.sudoku != -1)
    #     if (len(full_idx) == 0): return self # Nothing to pluck
    #     print("num to pluck", min(len(full_idx[0]), count))
    #     pluck_idx = np.random.choice(len(full_idx[0]), min(len(full_idx[0]), count), replace=False)
    #     pluck_row, pluck_col = full_idx[0][pluck_idx], full_idx[1][pluck_idx]
    #     print(pluck_row)
    #     print(pluck_col)

    #     res = SudokuCNF(self.order, self.sudoku)
    #     res.sudoku[[pluck_row], [pluck_col]] = -1
    #     return res

class SudokuSampler(SamplerBase):
    def __init__(self, order=3, num_filled=None, swaps=0, seed=None, use_classic=True, newspaper=False, **kwargs):
        SamplerBase.__init__(self, **kwargs)
        # if seed is None:
        #     random.seed(os.getpid())
        #     seed = int(time.time()) + random.randint(1,10000000)
        # else:
        #     seed = int(seed)
        # random.seed(seed)
        # np.random.seed(seed)

        self.use_classic = use_classic
        self.newspaper = newspaper
        self.order = int(order)
        self.size = self.order**2
        self.num_filled = int(num_filled) if (num_filled) else self.size
        self.swaps = int(swaps)

    def __rand_full_puzzle(self):
        while(True):
            seed_puzzle = np.hstack((np.random.choice(self.size, self.order), [-1] * (self.size**2 - self.order)))
            np.random.shuffle(seed_puzzle)
            seed_puzzle = seed_puzzle.reshape((self.size, self.size))

            answer = SudokuCNF(self.order, seed_puzzle).solve()

            if (answer == None): # The seed_puzzle is not solvable. Try again!
                continue

            ######## UNCOMMENT FOR TESTING ########
            # fitted = answer.fit(glucose.get_model())
            # plucked = fitted.pluck(self.size**2 - self.num_filled)

            # answer.display()
            # print("--------")
            # fitted.display()
            # print("--------")
            # plucked.display()
            # print("========")
            ######## END OF TESTING ########

            return answer

    """
        Turn a random puzzle with only 1 solution (like the ones in the papers)
    """
    def __get_newspaper_puzzle(self, full_puzzle):
        from filters.sharpsat_filter import SharpSATFilter
        self.log.info("Sampling random puzzle with only 1 solution (like the ones in the papers.)")

        tmp_file = f'/tmp/sudoku_gen_{random_string(8)}.cnf'
        fltr = SharpSATFilter(steps_min=0, time_min=0, time_max=2000, count_max=1)

        puzzle = copy.deepcopy(full_puzzle)
        idx = np.mgrid[0:self.size, 0:self.size].reshape(2,-1).T
        np.random.shuffle(idx)
        num_plucked = 0
        pop = len(idx)
        stats = {}

        while (pop != 0):
            num_plucked += 1
            pop -= 1
            elem, idx = idx[pop], idx[:pop] # pop the last index

            val = puzzle.sudoku[elem[0], elem[1]] # remember the value of the cell before plucking
            puzzle.sudoku[elem[0], elem[1]] = -1
            puzzle.encode().print_dimacs(tmp_file)

            if (not fltr.filter(tmp_file, stats)): # more than one model (backtrack)
                puzzle.sudoku[elem[0], elem[1]] = val
                num_plucked -= 1

        return puzzle, num_plucked

    def sample_board(self, stats_dict: dict = {}) -> SudokuCNF:
        full_puzzle = self.__rand_full_puzzle()
        if (self.newspaper):
            puzzle, num_plucked = self.__get_newspaper_puzzle(full_puzzle)
            self.num_filled = self.size**2 - num_plucked
            stats_dict.update({'num_plucked': num_plucked})
        else:
            puzzle = full_puzzle.pluck(self.size**2 - self.num_filled)
            # print("Before")
            # puzzle.display()
            puzzle = puzzle.random_swap(self.swaps)
            # print("After")
            # puzzle.display()

        return puzzle

    def sample(self, stats_dict: dict = {}) -> (FileName, FileName):
        puzzle = self.sample_board(stats_dict)
        suffix = random_string(8)
        cnf_id = "sudoku-%ix%i-%i-%s" % (self.size, self.size, self.num_filled, suffix)
        fname = os.path.join("/tmp", f"{cnf_id}.cnf")
        stats_dict.update({
            'file': cnf_id,
            'order': self.order,
            'num_filled': self.num_filled
        })

        puzzle.encode().print_dimacs(fname)

        self.log.info(f"Sampled {cnf_id} with {self.swaps} swaps.")

        return fname, None