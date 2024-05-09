import os
from subprocess import DEVNULL, STDOUT, check_call, check_output
import random
import string
import itertools

from gen_types import FileName
from samplers.sampler_base import SamplerBase
from gen_utils import random_string

class SHASampler(SamplerBase):
    def __init__(self, str_len_min=6, str_len_max=6, var_min=20, var_max=40, rounds_min=10, rounds_max=20, seed=None, **kwargs):
        SamplerBase.__init__(self, **kwargs)
        self.str_len_min = int(str_len_min)
        self.str_len_max = int(str_len_max) + 1
        self.var_min = int(var_min)
        self.var_max = int(var_max) + 1
        self.rounds_min = int(rounds_min)
        self.rounds_max = int(rounds_max) + 1

        self.param_gen = itertools.cycle(itertools.product(range(self.str_len_min, self.str_len_max), range(self.var_min, self.var_max), range(self.rounds_min, self.rounds_max)))
        random.seed(seed)

    def sample(self, stats_dict: dict) -> FileName:
        params = self.param_gen.__next__()

        str_len = params[0]
        str_arg = random_string(str_len)
        exc_arg = params[1]
        rounds_arg = params[2]

        cnf_id = "SHA-%i-%i-%i-%s" % (str_len, exc_arg, rounds_arg, random_string(8))
        fname = os.path.join("/tmp", f"{cnf_id}.cnf")

        command = f"cgen encode SHA1 -vM string:{str_arg} pad:sha1 except:{exc_arg} -vH random -r {rounds_arg} {fname}"
        # os.system(command)
        try:
            check_output(command.split(' '))#, stdout=STDOUT, stderr=STDOUT)
        except:
            raise Exception("cgen failure!")


        stats_dict.update({
            'file': cnf_id,
            'str_len': str_len,
            'var': exc_arg,
            'round': rounds_arg
        })

        self.log.info(f"Sampled {cnf_id}")

        return fname, None


    # def sample(self, stats_dict: dict) -> FileName:
    #     str_len = random.randrange(self.str_len_min, self.str_len_max)
    #     str_arg = random_string(str_len)
    #     exc_arg = random.randrange(self.var_min, self.var_max)

    #     cnf_id = "SHA-%i-%i-%s-%s" % (str_len, exc_arg, str_arg, random_string(8))
    #     fname = os.path.join("/tmp", f"{cnf_id}.cnf")

    #     command = f"cgen encode SHA1 -vM string:{str_arg} pad:sha1 except:{exc_arg} -vH random -r {self.rounds} {fname}"
    #     # os.system(command)
    #     try:
    #         check_output(command.split(' '))#, stdout=STDOUT, stderr=STDOUT)
    #     except:
    #         raise Exception("cgen failure!")


    #     stats_dict.update({
    #         'file': cnf_id,
    #         'str_len_min': self.str_len_min,
    #         'str_len_max': self.str_len_max,
    #         'var_min': self.var_min,
    #         'var_max': self.var_max
    #     })

    #     self.log.info(f"Sampled {cnf_id}")

    #     return fname, None
