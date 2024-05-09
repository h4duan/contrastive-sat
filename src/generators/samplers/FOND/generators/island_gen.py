import os
import sys
import numpy as np
import random
import itertools
from gen_utils import random_string


def gen(N, M, fname):
    task = open(fname, 'w')

    task.write('(define (problem islands-0)')
    task.write('\n')
    task.write('(:domain islands)')
    task.write('\n')
    task.write('(:objects ')
    # task.write('\th - home')
    task.write
    task.write('\n')
    for i in range(N):
        for j in range(N):
            task.write('\tL%i%i-1 - location' % (i + 1, j + 1))
            task.write('\n')
    task.write('\n')
    for i in range(N):
        for j in range(N):
            task.write('\tL%i%i-2 - location' % (i + 1, j + 1))
            task.write('\n')
    task.write('\n')
    for i in range(M):
        task.write('\tm%i - monkey' % (i + 1))
        task.write('\n')
    task.write(')')
    task.write('\n')

    task.write('(:init')
    task.write('\n')

    task.write('\t(person-alive)\n')
    task.write('\t(person-at L%i%i-1)\n' % (N, N))
    task.write('\t(bridge-clear)\n')
    task.write('\t; If some monkey initially in bridge change this!\n')
    task.write('\n')

    task.write('\t(bridge-drop-location L11-1)\n')
    task.write('\t(bridge-drop-location L11-2)\n')
    task.write('\n')
    for i in range(N):
        task.write('\t(swim-road L%i%i-1 L%i1-2) (swim-road L%i1-2 L%i%i-1)\n' % (i + 1, N, N, N, i + 1, N))
    task.write('\n')
    for i in range(N):
        task.write('\t(bridge-road L%i1-1 L%i%i-2) (bridge-road L%i%i-2 L%i1-1)\n' % (i + 1, i + 1, N, i + 1, N, i + 1))

    task.write('\n')
    deltaF = [-1, 1, 0, 0]
    deltaC = [0, 0, 1, -1]
    for i in range(N):
        for j in range(N):
            f = i + 1
            c = j + 1
            for df, dc in zip(deltaF, deltaC):
                x = f + df
                y = c + dc
                if x <= 0 or y <= 0 or x > N or y > N:
                    continue
                else:
                    task.write('\t(road L%i%i-1 L%i%i-1)' % (f, c, x, y))
                    task.write('\n')

    task.write('\n')
    for i in range(N):
        for j in range(N):
            f = i + 1
            c = j + 1
            for df, dc in zip(deltaF, deltaC):
                x = f + df
                y = c + dc
                if x <= 0 or y <= 0 or x > N or y > N:
                    continue
                else:
                    task.write('\t(road L%i%i-2 L%i%i-2)' % (f, c, x, y))
                    task.write('\n')

    task.write('\n')
    for i in range(M):
        task.write('\t(monkey-at m%i L%i%i-%i)\n' % (i + 1, np.random.randint(N) + 1, np.random.randint(N) + 1, np.random.randint(2) + 1))

    task.write('\t;Change monkeys location at will\n')

    task.write(')')
    task.write('\n')

    task.write('(:goal (person-at L%i1-2))' % (N))
    task.write('\n')
    task.write(')')

    task.close()


class IslandGen():
    def __init__(self, N_min=2, N_max=2, M_min=2, M_max=2, **kwargs):
        N_range = range(int(N_min), int(N_max) + 1)
        M_range = range(int(M_min), int(M_max) + 1)

        self.ranges = list(itertools.product(N_range, M_range))

    def sample(self): #TODO: Add state_dict
        (N, M) = random.choice(self.ranges)
        problem_id = "island-%i-%i-%s" % (N, M, random_string(8))
        fname = os.path.join("/tmp", f"{problem_id}.pddl")

        gen(N, M, fname)
        return problem_id, fname

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Wrong usage, you should use python name.py N M fname...')
        exit()

    N = int(sys.argv[1]) # size of the sides of the sqare matrix
    M = int(sys.argv[2])
    nameTask = sys.argv[3]

    gen(N, M, nameTask)
