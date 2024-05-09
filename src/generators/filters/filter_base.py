import logging

from gen_types import FileName

class FilterBase:
    def __init__(self, **kwargs):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')

        self.log = logging.getLogger(__name__)

    def filter(self, fname: FileName, stats_dict: dict) -> bool:
        raise NotImplementedError


class TrueFilter(FilterBase):
    def filter(self, fname: FileName, stats_dict: dict) -> bool:
        return True

class FalseFilter(FilterBase):
    def filter(self, fname: FileName, stats_dict: dict) -> bool:
        return False
