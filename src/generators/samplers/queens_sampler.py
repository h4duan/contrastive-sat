import os
import numpy as np
from subprocess import DEVNULL, STDOUT, check_call, check_output
import random
import string
import itertools

from gen_types import FileName
from samplers.sampler_base import SamplerBase
from gen_utils import random_string

class QueensSampler(SamplerBase):
    def __init__(self, size=8, blocks=4, seed=None, **kwargs):
        SamplerBase.__init__(self, **kwargs)
        self.size = int(size)
        self.blocks = int(blocks)
        self.nvars = self.size ** 2
        self.clauses = []
        # self.clausesH = []

        # Each row contains a queen
        self.clauses.extend([self.var(i, j) for j in range(self.size)] for i in range(self.size))

        # Prevent Attacking
        for i in range(self.size):
            for j in range(self.size):
                # row attack
                self.clauses.extend([-self.var(i, j), -self.var(i, k)] for k in range(j + 1, self.size))

                # column attack
                self.clauses.extend([-self.var(i, j), -self.var(k, j)] for k in range(i + 1, self.size))

                # + diagonal attack
                self.clauses.extend([-self.var(i, j), -self.var(i + k, j + k)] for k in range(1, self.size - max(i, j)))
                # self.clausesH.extend([(-(i + 1), -(j + 1)), (-(i + k + 1), -(j + k + 1))] for k in range(1, self.size - max(i, j)))

                # - diagonal attack
                self.clauses.extend([-self.var(i, j), -self.var(i - k, j + k)] for k in range(1, min(i + 1, self.size - j)))
                # self.clausesH.extend([(-(i + 1), -(j + 1)), (-(i - k + 1), -(j + k + 1))] for k in range(1, min(i + 1, self.size - j)))

        # print(self.clauses)
        # print(len(self.clauses))

        random.seed(seed)

    def var(self, i, j):
        return i * self.size + j + 1

    def print_dimacs(self, units, fname = None):
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

        f.write("c\nc CNF Blocked N-Queens Translation\nc\np cnf " + str(self.nvars) + " "
                + str(len(self.clauses) + len(units))+"\n")
        for clause in self.clauses:
            print_dimacs_clause(clause)
        for unit in units:
            print_dimacs_clause(unit)

        if fname != None: f.close()

    def sample(self, stats_dict: dict) -> FileName:
        cnf_id = "queens-%i-%i-%s" % (self.size, self.blocks, random_string(8))
        fname = os.path.join("/tmp", f"{cnf_id}.cnf")

        units = -np.random.choice(range(1, self.nvars + 1), self.blocks, replace=False).reshape((-1, 1))

        self.print_dimacs(units, fname)

        stats_dict.update({
            'file': cnf_id,
            'size': self.size,
            'blocks': self.blocks
        })

        self.log.info(f"Sampled {cnf_id}")

        return fname, None
