import os
from pathlib import Path

def power_law(out_file, seed):
    # sgen1 -n num-of-variables [-sat | -unsat] [-s random-seed] [-m satisfying-model-file] [-min-variables] [-reorder]
    command = f'./CreateSAT.bin -g p -v 10 -c 41 -k 3 -p 1.7 -f {out_file} -u 1 -s {seed}'
    os.system(f'{command} > output')
