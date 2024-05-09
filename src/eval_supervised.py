
import os


import torch
import torch.nn as nn
import torch.optim as optim

from torch import autograd

from neurosat import NeuroSAT
from neurosatsimp import NeuroSATSimp

from config import parser

from loss import supervised
from time import localtime, strftime

from sat_dataset import SATInstances
from data_maker import generate_train
from torch.utils.data import DataLoader

from sat_dataset import collate_fn_test

import wandb

import pickle

args = parser.parse_args()

net = None
if args.neurosat:
  net = NeuroSAT(args)
elif args.neurodiver:
  net = NeuroSATSimp(args)
else:
  print("ERROR: Pick at least one model")
  quit

net = net.cuda()
#if args.model_file != "":
    #net.load_state_dict(torch.load(args.model_file))


time = strftime("%Y-%m-%d %H:%M:%S", localtime()).replace(" ", "_")
task_name = args.task_name  + '_' + time +   '_lr' + str(args.learning_rate) + '_label' + str(args.label_proportion)

log_file = open(os.path.join(args.log_dir, task_name+'.log'), 'a+')

for arg in vars(args):
  if args.screen:
    print (arg, getattr(args, arg))
  else:
    print (arg, getattr(args, arg), flush = True, file = log_file)


sigmoid  = nn.Sigmoid()

print_screen = args.screen

best_acc = 0.0
start_epoch = 0

def print_grad(model):
    grads = []
    for name, param in model.named_parameters():
        if param.grad != None:
            grads.append(param.grad.view(-1))
        #grads.append(param.grad.view(-1))
        #print(name, param.grad)
        #print(name)
        #print(torch.count_nonzero(param.grad) / torch.numel(param.grad))
        #print(" ")
        #grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    #print("non zero " + str(torch.count_nonzero(grads) / len(grads)))
    print(torch.sort(torch.absolute(grads)))
    #print(len(grads))

if not args.debug:
    wandb.login(key="49fcdbf0a6500043ff9402a8555c64a7351364d0")
    wandb.init(project='supervised', entity='sslsat', name = task_name)
    wandb.config.update(args)
    wandb.config.jobid = os.environ["SLURM_JOB_ID"]


jobid = None

if not args.debug:
    jobid = os.environ["SLURM_JOB_ID"]


labels = None
n_vars = None
label_file = os.path.join(args.val_folder, args.val_label)
with open(label_file, 'rb') as handle:
  labels = pickle.load(handle)

n_vars_file = os.path.join(args.val_folder, args.val_n_vars)
with open(n_vars_file, 'rb') as handle:
  n_vars = pickle.load(handle)

#labels = labels[:5000]
#n_vars = n_vars[:5000]

#train_size = int(len(labels) * args.label_proportion)
#if not args.debug:
#    wandb.config.train_size = train_size
#print("size of training set: " + str(train_size))
#test_size = min(len(labels) - train_size, 2000)

#train_labels = labels[test_size:test_size+train_size]
#train_n_vars = n_vars[test_size:test_size+train_size]
#test_labels = labels[:test_size]
#test_n_vars = n_vars[:test_size]



val_dataset = SATInstances(args, start=0, data_length=args.test_size, labels=labels, n_vars=n_vars, augment = False)
val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, collate_fn = collate_fn_test, shuffle =  True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

supervised_train = supervised(args.dim)
supervised_train = supervised_train.cuda()

net.load_state_dict(torch.load(os.path.join(args.model_dir, "net_"+args.job_id+".pth")))
supervised_train.load_state_dict(torch.load(os.path.join(args.model_dir, "supervised_train_"+args.job_id+".pth")))
#optim = optim.Adam(list(net.parameters()) + list(supervised_train.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
#best_acc = 0
#train1 = generate(args)
net.eval()
supervised_train.eval()

    #optim.zero_grad()
total = 0
correct = 0
for prob in val_dataloader:
    outputs = net(prob)
      #print(outputs)
    target = torch.Tensor(prob.is_sat).cuda().long()
    total_cur, correct_cur = supervised_train.evaluate(outputs, target)
    total += total_cur
    correct += correct_cur
accuracy = correct / total
    #print(total)
#if not args.debug:
print("test accuracy: " + str(accuracy))

