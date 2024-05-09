import os
from pathlib import Path

def sgen(num_vars, out_dir, prob_kind, seed, sat_model_file=None, min_vars=False, reorder=True):
    # sgen1 -n num-of-variables [-sat | -unsat] [-s random-seed] [-m satisfying-model-file] [-min-variables] [-reorder]
    command = f'./sgen1 -n {num_vars} -{prob_kind.lower()} -s {seed}'
    if sat_model_file:
        command += f' -m {sat_model_file}'
    if min_vars:
        command += ' -min-variables'
    if reorder:
        command += ' -reorder'

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cnf_file = f'sgen-{prob_kind.lower()}-{num_vars}-{seed}.cnf'
    os.system(f'{command} > {out_dir}/{cnf_file}')
    #print(f'{command} > {cnf_file}')


#sgen(100, "dataset")
#sgen(90, "dataset", "UNSAT")
#sgen(80, "dataset", "UNSAT")
#sgen(70, "dataset", "UNSAT")
#sgen(60, "dataset", "UNSAT")
#sgen(50, "dataset", "UNSAT")
