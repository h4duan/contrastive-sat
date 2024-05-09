import os
import numpy as np
from subprocess import DEVNULL, STDOUT, check_call, check_output
import random
import time
import string
import itertools
from contextlib import suppress

from gen_types import FileName
from samplers.sampler_base import SamplerBase

from samplers.FOND.pddl_to_cnf import toCNF
from samplers.FOND.generators.island_gen import *

class FONDSampler(SamplerBase):
    def __init__(self, domain, state_cnt = 1, seed=None, **kwargs):
        SamplerBase.__init__(self, **kwargs)

        if seed is None:
            random.seed(os.getpid())
            seed = int(time.time()) + random.randint(1,10000000)
        else:
            seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)


        self.pddl_domain = os.path.join("samplers", "FOND", "domains", domain + "-domain.pddl")
        self.state_cnt = int(state_cnt)
        if domain == 'island':
            self.generator = IslandGen(**kwargs)
        else:
            assert False, 'Wrong FOND_GEN!!!'



    def sample(self, stats_dict: dict) -> FileName:
        problem_id, pddl_file = self.generator.sample()
        cnf_file = os.path.join("/tmp", f"{problem_id}.cnf")
        toCNF(self.pddl_domain, pddl_file, cnf_file, self.state_cnt)
        with suppress(FileNotFoundError):
                os.remove(pddl_file)

        self.log.info(f"Sampled {problem_id}")

        return cnf_file, None
