import os
from pathlib import Path

def double_power_law(out_file, seed):
    # sgen1 -n num-of-variables [-sat | -unsat] [-s random-seed] [-m satisfying-model-file] [-min-variables] [-reorder]
    #./CreateSAT -g d -v 20 -c 36 -k 4 -p 1.7 -f "test_$i" -u 1 -s $RANDOM
    command = f'./CreateSAT.bin -g d -v 20 -c 36 -k 4 -p 1.7 -f {out_file} -u 1 -s {seed}'
    os.system(f'{command} > output')
