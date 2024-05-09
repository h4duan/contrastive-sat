import os
import random
import itertools

from gen_types import FileName
from samplers.sampler_base import SamplerBase
from gen_utils import random_string

class Rule:
    def __init__(self, rule_id):
        self.rule_id = rule_id

        clauses = set()

        for i in range(8):
            output = (rule_id >> i) & 1
            in_l = (i >> 2) & 1
            in_c = (i >> 1) & 1
            in_r = (i >> 0) & 1

            clauses.add((not in_l, not in_c, not in_r, output))

        minimal_clauses = set()

        while clauses:
            next_clauses = set()
            for clause in clauses:
                is_minimal = True
                for i, c_i in enumerate(clause):
                    if c_i != 2:
                        flipped = list(clause)
                        flipped[i] = not c_i
                        if tuple(flipped) in clauses:
                            flipped[i] = 2
                            next_clauses.add(tuple(flipped))
                            is_minimal = False
                if is_minimal:
                    minimal_clauses.add(clause)
            clauses = next_clauses

        self.template = sorted(minimal_clauses)

    def clauses(self, *literals):
        for clause in self.template:
            yield [
                [-lit, lit][pol]
                for pol, lit in zip(clause, literals)
                if pol != 2
            ]

    def apply_cell(self, l, c, r):
        return bool((self.rule_id >> ((l << 2) + (c << 1) + (r << 0))) & 1)

    def apply(self, state):
        size = len(state)
        return [
            self.apply_cell(
                state[(i - 1) % size],
                state[i],
                state[(i + 1) % size]
            ) for i in range(size)
        ]


class Benchmark:
    def __init__(self, rule, state_size, fwd_steps, rev_steps, seed):
        self.rule = Rule(rule)
        self.state_size = state_size
        self.fwd_steps = fwd_steps
        self.rev_steps = rev_steps
        self.seed = seed

        self.variable_count = 0
        self.clauses = []

        final_state_values = self.generate_final_state()
        final_state_variables = self.generate_transitions()

        for value, variable in zip(final_state_values, final_state_variables):
            if value:
                self.clauses.append([variable])
            else:
                self.clauses.append([-variable])

    @property
    def name(self):
        return 'ecarev-%i-%i-%i-%i-%i' % (
            self.rule.rule_id, self.state_size, self.fwd_steps,
            self.rev_steps, self.seed
        )

    def make_variable(self):
        self.variable_count += 1
        return self.variable_count

    def generate_final_state(self):
        rng = random.Random()
        rng.seed(self.name, version=2) # make sure its random

        state = [rng.getrandbits(1) for x in range(self.state_size)]

        for t in range(self.fwd_steps):
            state = self.rule.apply(state)

        return state

    def generate_transitions(self):
        state = [self.make_variable() for x in range(self.state_size)]

        for t in range(self.rev_steps):
            next_state = [self.make_variable() for x in range(self.state_size)]

            for x in range(self.state_size):
                pl = state[(x - 1) % self.state_size]
                pc = state[x]
                pr = state[(x + 1) % self.state_size]
                c = next_state[x]

                self.clauses.extend(self.rule.clauses(pl, pc, pr, c))

            state = next_state

        return state

    def print_dimacs(self, file=None):
        print('c', self.name, file=file)
        print('p cnf', self.variable_count, len(self.clauses), file=file)
        for clause in self.clauses:
            print(' '.join(map(str, clause + [0])), file=file)

class EcarevSampler(SamplerBase):
    def __init__(self,  R_min = 37, R_max = 38,
                        n_min = 100, n_max = 101, n_inc = 1,
                        f_min = 11, f_max = 40, f_inc = 1,
                        r_min = 20, r_max = 40, r_inc = 1, seed=None, **kwargs):
        SamplerBase.__init__(self, **kwargs)
        R_range = range(int(R_min), int(R_max) + 1)
        n_range = range(int(n_min), int(n_max) + 1, int(n_inc))
        f_range = range(int(f_min), int(f_max) + 1, int(f_inc))
        r_range = range(int(r_min), int(r_max) + 1, int(r_inc))
        random.seed(seed)

        self.ranges = list(itertools.product(R_range, n_range, f_range, r_range))

    def sample(self, stats_dict: dict) -> FileName:

        (R, n, f, r) = random.choice(self.ranges)
        s = random.randrange(2, pow(2, n))
        cnf_id = "ecarev-%i-%i-%i-%i-%s" % (R, n, f, r, random_string(8))

        fname = os.path.join("/tmp", f"{cnf_id}.cnf")
        Benchmark.print_dimacs(Benchmark(R, n, f, r, s), open(fname, "w"))
        stats_dict.update({
            'file': cnf_id,
            'rule': R,
            'n': n,
            'f': f,
            'r': r,
            's': s
        })

        self.log.info(f"Sampled {cnf_id}")

        return fname, None
