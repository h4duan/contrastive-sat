import os, sys
import ray
import csv
import argparse
import shutil
from contextlib import suppress

from filters.filter_base import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
log = logging.getLogger(__name__)

def get_sampler(config):
    if config['sampler'] == 'word':
        from samplers.word_sampler import WordSampler

        return WordSampler(**config)
    elif config['sampler'] == 'grid':
        from samplers.grid_sampler import GridSampler

        return GridSampler(**config)
    elif config['sampler'] == 'sudoku':
        from samplers.sudoku_sampler import SudokuSampler

        return SudokuSampler(**config)
    elif config['sampler'] == 'ecarev':
        from samplers.ecarev_sampler import EcarevSampler

        return EcarevSampler(**config)
    elif config['sampler'] == 'sha':
        from samplers.sha_sampler import SHASampler

        return SHASampler(**config)
    elif config['sampler'] == 'queens':
        from samplers.queens_sampler import QueensSampler

        return QueensSampler(**config)
    elif config['sampler'] == 'fond':
        from samplers.FOND_sampler import FONDSampler

        return FONDSampler(**config)
    else:
        assert False, 'WHHAAT?'

def get_filter(config):
    if config['filter'] == 'sharp':
        print("glucoseglucoseglucoseglucose")
        from filters.sharpsat_filter import SharpSATFilter

        return SharpSATFilter(**config)
    elif config['filter'] == 'glucose':
        from filters.glucose_filter import GlucoseFilter

        return GlucoseFilter(**config)
    elif config['filter'] == 'true':
        return TrueFilter(**config)
    elif config['filter'] == 'false':
        return FalseFilter(**config)
    else:
        assert False, 'WHHAAT?'

@ray.remote
def generate_from_sampler(config):
    sampler = get_sampler(config)
    fltr = get_filter(config)
    stats_dict = {}
    attempts_so_far = 0
    while (attempts_so_far < config['max_attempts']):
        try:
            attempts_so_far += 1
            candidate, extra_data = sampler.sample(stats_dict)
        except Exception as e:
            log.error('Gah, Exception:')
            log.error(e, exc_info=True)
            stats_dict = {}
            continue

        if fltr.filter(candidate, stats_dict):
            with suppress(FileNotFoundError):
                shutil.move(candidate,os.path.join(config['dir'], os.path.basename(candidate)))
                if extra_data is not None:
                    shutil.move(extra_data,os.path.join(config['dir'], os.path.basename(extra_data)))
            break
        else:
            with suppress(FileNotFoundError):
                os.remove(candidate)
                if extra_data is not None:
                    os.remove(extra_data)
            stats_dict = {}

    return stats_dict

def generate_dataset(args):
    assert args.n, "Must define target number"
    dst = args.destination_dir
    config = {
        'sampler': args.sampler,
        'filter': args.filter,
        'dir': dst,
        'max_attempts': args.max_attempts
    }
    for p in args.params:
        k, v = p.split('=')
        config[k]=v
    with suppress(OSError):
        os.mkdir(dst)

    results = [generate_from_sampler.remote(config) for _ in range(args.n)]
    stats = ray.get(results)

    if (stats[0]): # if the dicts are not empty
        with open(os.path.join(config['dir'], "stats.csv"), 'a', newline='') as statsfile:
            writer = csv.DictWriter(statsfile, fieldnames=stats[0].keys())
            writer.writeheader()
            for stat in stats:
                writer.writerow(stat)

    log.info('Finished')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reject Sampling for CNF files.')
    parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
    parser.add_argument('-d', '--destination_dir', type=str, default=os.curdir, help='destination directory')
    parser.add_argument('-s', '--sampler', type=str, default='word', help='Sampler (generator)')
    parser.add_argument('-f', '--filter', type=str, default='sharp', help='Filter')
    parser.add_argument('-n', type=int, default=0, help='Number of formulas to generate')
    parser.add_argument('-m', '--max_attempts', type=int, default=sys.maxsize, help='Max number of generation attempts by a worker.')
    parser.add_argument('-p', '--parallelism', type=int, default=1, help='number of cores to use (Only if not in cluster mode)')
    parser.add_argument('-c', '--cluster', action='store_true', default=False, help='run in cluster mode')
    args = parser.parse_args()
    if args.cluster:
        ray.init(address='auto', redis_password='blabla')
    else:
        ray.init(num_cpus=args.parallelism)

    generate_dataset(args)
