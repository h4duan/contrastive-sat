import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', type=str, default='neurosat', help='task name')

parser.add_argument('--dim', type=int, default=128, help='Dimension of variable and clause embeddings')
parser.add_argument('--n_rounds', type=int, default=26, help='Number of rounds of message passing')
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--min_n', type=int, default=10, help='min number of variables used for training')
parser.add_argument('--max_n', type=int, default=40, help='max number of variables used for training')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)
parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)
parser.add_argument('--one', action='store', dest='one', type=int, default=0)
parser.add_argument('--num_batch', type=int, default=1)
parser.add_argument('--log_dir', type=str, help='log folder dir')
parser.add_argument('--model_file', type=str, help='model file')
parser.add_argument('--data_dir', type=str, help='data folder dir')
parser.add_argument('--restore', type=str, default=None, help='continue train from model')
parser.add_argument('--weight_decay', type=float, default=1e-10)


parser.add_argument('--train-file', type=str, default=None, help='train file dir')
parser.add_argument('--val_folder', type=str, default=None, help='val file dir')
parser.add_argument('--val_label', type=str, default=None, help='val label file')
parser.add_argument('--val_n_vars', type=str, default=None, help='val n_vars file')
parser.add_argument('--val_instance', type=str, default=None, help='val var file name')
parser.add_argument('--val_num_data', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--label_proportion', type=float)

#parser.add_argument('--supervised_augment', type=bool, help = 'supervised augmentation')
parser.add_argument('--print-screen', dest = "screen", action = "store_true")
parser.add_argument('--supervised_augment', dest = "augment", action = "store_true")
parser.add_argument('--ssl', dest = "ssl", action = "store_true")


parser.add_argument('--no-print-screen', dest = "screen", action = "store_false")
parser.add_argument('--compute-train-accuracy', dest = 'train_accuracy', action = "store_true")


parser.set_defaults(screen=False)
parser.set_defaults(augment=False)
parser.set_defaults(ssl=False)
parser.set_defaults(train_accuracy=False)
parser.set_defaults(simclr=False)
parser.set_defaults(vicreg=False)
