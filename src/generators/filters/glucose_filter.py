import os

from filters.filter_base import FilterBase
from gen_types import FileName

from pysat.solvers import Glucose3
from pysat.formula import CNF

"""
Filters the SAT instance according to a set of constrainst:
    steps_min
    time_min
    time_max

sample usage:
>>> # import GlucoseFilter
>>> file_path = # path to the cnf file
>>> time_max = 10
>>> filter = GlucoseFilter()
>>> filter.filter(file_path)
"""
class GlucoseFilter(FilterBase):
    def __init__(self, steps_min = 4, steps_max = 100, op_cnt_max = 1e10, time_min = 0, time_max = 10, reduce_base=2000, max_var=-1, **kwargs):
        FilterBase.__init__(self, **kwargs)
        self.steps_min = int(steps_min)
        self.steps_max = int(steps_max)
        self.op_cnt_max = int(op_cnt_max)
        self.time_min = float(time_min)
        self.time_max = int(time_max)
        self.reduce_base = int(reduce_base)
        self.max_var = int(max_var)

    def filter(self, fname: FileName, stats_dict: dict) -> bool:
        cnf = CNF()
        cnf.from_file(fname)

        # glucose = Glucose3(gc_oracle = {"callback": lambda *args: None, "policy":"glucose"}, gc_freq="fixed", reduce_base=self.reduce_base, use_timer=True)
        glucose = Glucose3(use_timer=True)
        glucose.append_formula(cnf.clauses)
        # glucose.time_budget(self.time_max)
        solver_result = glucose.solve_limited()

        message = ""
        res = True
        # if (self.op_cnt_max < glucose.reward()):
        #     message = f"{fname}: Too hard: op_cnt > {self.op_cnt_max}"
        #     res = False
        # if (glucose.nof_gc() < self.steps_min):
        #     message = f"{fname}: Degenerate: nof_gc < {self.steps_min}"
        #     res = False
        # if (self.steps_max < glucose.nof_gc()):
        #     message = f"{fname}: Too many steps: nof_gc > {self.steps_max}"
        #     res = False
        if (len(cnf.clauses) == 0):
            message = f"{fname}: Empty file, perhaps solved by UP while sampling."
            res = False
        if (glucose.time() < self.time_min):
            message = f"{fname}: Too easy! Time < {self.time_min}s"
            res = False
        if (self.time_max <= glucose.time()):
            message = f"{fname}: Too hard: Time > {self.time_max}s"
            res = False
        if (self.max_var > 0 and cnf.nv > self.max_var):
            message = f"{fname}: Too many vars ({cnf.nv} > {self.max_var})"
            res = False

        if (res): # Instance accepted. Save the stats about the new instance
            message = f"{fname}: Accepted"
            stats_dict.update({
                'var_len': glucose.nof_vars(),
                'cla_len': glucose.nof_clauses(),
                # 'gc_cnt' : glucose.nof_gc(),
                # 'op_cnt' : glucose.reward(),
                'time'   : f"{glucose.time():.2f}",
                'result' : solver_result
            })

        message += f" (time/pid: {glucose.time():.2f}/{os.getpid()})"
        self.log.info(message)
        glucose.delete()

        return res
