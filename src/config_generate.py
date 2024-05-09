import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_n', type=int, default=10, help='min number of variables used for training')
parser.add_argument('--max_n', type=int, default=40, help='max number of variables used for training')
parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)
parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)


parser.add_argument('--val_folder', type=str, default=None, help='val file dir')
parser.add_argument('--val_label', type=str, default=None, help='val label file')
parser.add_argument('--val_n_vars', type=str, default=None, help='val n_vars file')
parser.add_argument('--val_instance', type=str, default=None, help='val var file name')
parser.add_argument('--val_num_data', type=int, default=None)
