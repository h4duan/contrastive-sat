#!/usr/bin/env python3

import os
import re
# from IPython.core.debugger import Tracer
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import io

MAX_CLAUSES_PER_VARIABLE = 30000

def is_number(s):
    try:
        int(s)
    except ValueError:
        return False
    return True

def clause_to_string(c):
    return ' '.join(map(str,c)) + ' 0\n'

def sign(x):
    return x and (1, -1)[x < 0]

def write_to_string(maxvar, clause_list, universals=set()):
    output = io.StringIO()
    output.write('p cnf {} {}\n'.format(maxvar,len(clause_list)))

    if len(universals) > 0:
        output.write('a')
        for u in universals:
            output.write(' {}'.format(u))
        output.write(' 0\n')
        output.write('e')
        for v in range(1,maxvar+1):
            if v not in universals:
                output.write(' {}'.format(v))
        output.write(' 0\n')
    # output.write('p cnf ' + str(maxvar) + ' ' + str(len(clause_list)) + '\n')
    for c in clause_list:
        output.write(clause_to_string(c))
    rc = output.getvalue()
    output.close()
    return rc

def write_to_file(maxvar, clause_list, filename, universals=set()):
    textfile = open(filename, "w")
    textfile.write(write_to_string(maxvar,clause_list,universals=universals))
    textfile.close()



def extract_num_conflicts(s):
    res = re.findall(' Conflicts: (\d+)', str(s))
    if len(res) == 1:
        return int(res[0])
    else:
        print('  ERROR: {}'.format(s))
        return 0


def extract_num_decisions(s):
    res = re.findall(' Decisions: (\d+)', str(s))
    if len(res) == 1:
        return int(res[0])
    else:
        print('  ERROR: {}'.format(s))
        return 0


def eval_formula(maxvar,clauses,universals=set(), repetitions=1):
    # print(str(maxvar))
    # print(str(clauses))
    # Tracer()()

    returncodes = []
    conflicts = []
    decisions = []

    for _ in range(repetitions):
        tool = ['./cadet','-v','1',
                '--debugging',
                '--cegar_soft_conflict_limit',
                '--sat_by_qbf',
                '--random_decisions',
                '--fresh_seed']
        p = Popen(tool,stdout=PIPE,stdin=PIPE)
        p.stdin.write(str.encode('p cnf {} {}\n'.format(maxvar,len(clauses))))
        if len(universals) > 0:
            p.stdin.write(str.encode('a'))
            for u in universals:
                p.stdin.write(str.encode(' {}'.format(u)))
            p.stdin.write(str.encode(' 0\n'))
            p.stdin.write(str.encode('e'))
            for v in range(1,maxvar+1):
                if v not in universals:
                    p.stdin.write(str.encode(' {}'.format(v)))
            p.stdin.write(str.encode(' 0\n'))
        for c in clauses:
            p.stdin.write(str.encode(clause_to_string(c)))
        stdout, stderr = p.communicate()
        # print('  ' + str(p.returncode))
        # print('  ' + str(stdout))
        # print('  ' + str(stderr))

        # print('  Maxvar: ' + str(maxvar))
        # print('  Conflicts: ' + str(extract_num_conflicts(stdout)))
        returncodes.append(p.returncode)
        conflicts.append(extract_num_conflicts(stdout))
        decisions.append(extract_num_decisions(stdout))

        p.stdin.close()

    assert all(x == returncodes[0] for x in returncodes)
    return (returncodes[0], np.mean(conflicts), np.mean(decisions))

def is_sat(maxvar,clauses,universals=set()):
    (returncode,_) = eval_formula(maxvar,clauses,universals)
    return returncode == 10

def dimacs_to_clauselist(dimacs):
    clauses = []
    assert(MAX_CLAUSES_PER_VARIABLE >= 8) # otherwise code below might not work
    maxvar = 0
    for line in dimacs:
        lits = line.split()[0:-1]
        if is_number(lits[0]):
            lits = list(map(int,lits))
            clauses.append(lits)
        elif lits[0] == b'p': # header
            assert(lits[1] == b'cnf')
            assert(is_number(lits[2]))
            maxvar = int(lits[2])
            assert(maxvar != 0)
        else: # must be comment in dimacs format
             if lits[0] != b'c':
                 print ('Could not read line ' + str(lits))
                 quit()
    assert (maxvar != 0)
    return maxvar, clauses

# def normalizeCNF_string(cnf):
#     occs = {}
#     maxvar = None
#     assert(MAX_CLAUSES_PER_VARIABLE >= 8) # otherwise code below might not work
#
#     for line in cnf:
#         lits = line.split()[0:-1]
#         if is_number(lits[0]):
#             lits = list(map(int,lits))
#             for l in lits:
#                 if abs(l) not in occs:
#                     occs[abs(l)] = []
#                 occs[abs(l)] += [lits] # storing reference to lits so we can
#                                        # manipulate them consistently
#         else:
#             if lits[0] == b'p':
#                 maxvar = int(lits[2])
#
#     assert(maxvar != None)
#
#     return normalizeCNF_occs(maxvar, occs)

def occs_to_clauses(maxvar, occs):
    clause_str_set = set()
    clauses = []
    for v in occs.keys():
        for c in occs[v]:
            c_string = ' '.join(map(str,c))
            if c_string not in clause_str_set: # because string is hashable
                clause_str_set.add(c_string)
                clauses.append(c)
    return maxvar, clauses

def normalizeCNF(clauses):
    occs = {}
    maxvar = None
    for lits in clauses:
        for l in lits:
            if abs(l) not in occs:
                occs[abs(l)] = []
            occs[abs(l)] += [lits] # storing reference to lits so we can manipulate them consistently

            if maxvar == None or maxvar < abs(l):
                maxvar = abs(l)
    return normalizeCNF_occs(maxvar, occs)

def normalizeCNF_occs(maxvar, occs):
    # Split variables when they occur in more than 8 clauses
    itervars = set(occs.keys())
    added_vars = 0
    while True:
        if len(itervars) == 0:
            break
        v = itervars.pop()
        if len(occs[v]) > MAX_CLAUSES_PER_VARIABLE:
            # print('Found var '+str(v)+ ' with '+ str(len(occs[v]))+ ' occurrences.')
            maxvar += 1
            added_vars += 1
            connector_clauses = [[v,-maxvar],[-v,maxvar]]
            # prepend connector_clauses to shift all clauses back, don't want
            # to remove the connector_clauses with what follows
            occs[v] = connector_clauses + occs[v]
            assert(len(occs[v][(MAX_CLAUSES_PER_VARIABLE+2):]) > 0)
                # > MAX_CLAUSES_PER_VARIABLE and we added two connector_clauses

            assert(maxvar not in occs)
            occs[maxvar] = connector_clauses

            # move surplus clauses over to new variable
            for clause in occs[v][MAX_CLAUSES_PER_VARIABLE:]:
                # change clause inplace, so change is consistent for occurrence lists of other variables
                clause[:] = list(map(lambda x: maxvar * sign(x) if abs(x) == v else x, clause))
                occs[maxvar] += [clause]
            assert(len(occs[v]) > len(occs[maxvar]))

            occs[v] = occs[v][:(MAX_CLAUSES_PER_VARIABLE - 1)]

            # if len(occs[maxvar]) > MAX_CLAUSES_PER_VARIABLE:
                # print('  new var '+str(maxvar)+ ' will be back.')

            itervars.add(maxvar)

    # print ('  maxvar: ' + str(maxvar))
    # print ('  added vars: ' + str(added_vars))
    # print ('  Max: ' + str( max( [len(occs[v]) for v in occs.keys()] )))
    # print ('  Over MAX_CLAUSES_PER_VARIABLE occs: '
        # + str(len( filter(lambda x: x, [len(occs[v])
        #   > MAX_CLAUSES_PER_VARIABLE for v in occs.keys()] ))))

    return occs_to_clauses(maxvar, occs)
