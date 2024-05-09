import argparse
import pickle
import os

from data_maker import gen_iclause_pair
from config_generate import parser
import random


args = parser.parse_args()

labels = []
n_vars = []
n_var_max = args.max_n
n_var_min = args.min_n
num_data = args.val_num_data
#problem_len = int(args.val_num_data / (args.max_n - args.min_n))

total_prob = 0

for i in range(num_data):
    print(i)
    n_var = random.randint(n_var_min, n_var_max)
    var, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(args, n_var)
    n_vars += [var]

    if random.random() < 0.5:
        iclauses.append(iclause_unsat)
        labels += [0]
    else:
        iclauses.append(iclause_sat)
        labels += [1]

    instance_name = os.path.join(args.val_folder, args.val_instance + "_" + str(total_prob) + ".pkl")
    total_prob += 1
    with open(instance_name, 'wb') as fp:
        pickle.dump(iclauses, fp)

label_file = os.path.join(args.val_folder, args.val_label)
with open(label_file, 'wb') as fp:
    pickle.dump(labels, fp)

n_vars_file = os.path.join(args.val_folder, args.val_n_vars)
with open(n_vars_file, 'wb') as fp:
    pickle.dump(n_vars, fp)







