import argparse
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT
from neurosatsimp import NeuroSATSimp
from data_maker import generate_train
from torch.utils.data import DataLoader

from sat_dataset import SATInstances
from sat_dataset import SATInstancesAugmentTwo
from sat_dataset import collate_fn_test
from sat_dataset import collate_fn_train

from config import parser
from loss import ssl
import random

from time import gmtime, strftime
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



args = parser.parse_args()

net = None
if args.neurosat:
    net = NeuroSAT(args)
if args.neurodiver:
    net = NeuroSATSimp(args)


net = net.cuda()
net.load_state_dict(torch.load(args.model_file))

time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "_")
task_name = args.task_name  + '_' + time +   '_lp' + str(args.label_proportion) 

log_file = open(os.path.join(args.log_dir, task_name+'.log'), 'a+')


for arg in vars(args):
  if args.screen:
    print (arg, getattr(args, arg))
  else:
    print (arg, getattr(args, arg), flush = True, file = log_file)


print(" ")


sigmoid  = nn.Sigmoid()

print_screen = args.screen

best_acc_lr = 0.0
best_acc_svm = 0.0
start_epoch = 0


val_dataset = None

if args.val_folder is None:
  print("no validation set")
  quit()

print("loading validation set")

labels = None
n_vars = None
label_file = os.path.join(args.val_folder, args.val_label)
with open(label_file, 'rb') as handle:
  labels = pickle.load(handle)
n_vars_file = os.path.join(args.val_folder, args.val_n_vars)
with open(n_vars_file, 'rb') as handle:
  n_vars = pickle.load(handle)


#labels = labels[:10000]
#n_vars = n_vars[:10000]

train_size = int(len(labels) * args.label_proportion)
print("train size " + str(train_size)) 
val_size = min(int((len(labels) - train_size)/2), 1000)
test_size = len(labels) - train_size - val_size





train_labels = labels[:train_size]
train_n_vars = n_vars[:train_size]
val_end = train_size + val_size
#print(train_size, val_end)
val_labels = labels[train_size:val_end]
val_n_vars = n_vars[train_size:val_end]
test_labels = labels[val_end:val_end + test_size]
test_n_vars = n_vars[val_end:val_end + test_size]




test_dataset = SATInstances(args, start=val_end, data_length=test_size, labels=test_labels, n_vars=test_n_vars, augment = False)
test_dataloader = DataLoader(test_dataset, shuffle = True, batch_size = 100, collate_fn = collate_fn_test, num_workers = args.num_workers)
val_dataset = SATInstances(args, start=train_size, data_length=val_size, labels=val_labels, n_vars=val_n_vars, augment = False)
val_dataloader = DataLoader(val_dataset, shuffle = True, batch_size = 100, collate_fn = collate_fn_test, num_workers = args.num_workers)
train_dataset = SATInstances(args, start=0, data_length=train_size, labels=train_labels, n_vars=train_n_vars, augment = False)
train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = 100, collate_fn = collate_fn_test, num_workers = args.num_workers)
print("finish loading")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(train_x, train_y, test_x, test_y, val_x, val_y):
  possible_c = [100000, 0.001, 1000, 100, 10, 1, 0.1, 0.01, 10000]
  best_val_lr = 0
  best_val_svm = 0
  test_lr = 0
  test_svm = 0
  train_x = np.asarray(train_x)
  train_y = np.asarray(train_y, dtype=int)
  val_x = np.asarray(val_x)
  val_y = np.asarray(val_y, dtype=int)
  test_x = np.asarray(test_x)
  test_y = np.asarray(test_y, dtype=int)
  for c in possible_c:
    clf_lr = LogisticRegression(max_iter=20000, C=c).fit(train_x, train_y)
    #clf_svm = SVC(gamma="scale", C=c).fit(train_x, train_y)
    local_lr = clf_lr.score(val_x, val_y)
    #print(local_svm)
    #local_svm = clf_svm.score(val_x, val_y)
    #print(local_svm, c)
    if local_lr > best_val_lr:
      best_val_lr = local_lr
      test_lr = clf_lr.score(test_x, test_y)
    #if local_svm > best_val_svm:
    #  best_val_svm = local_svm
    #  test_svm = clf_svm.score(test_x, test_y)
  return test_lr


net.eval()
train_x = []
train_y = []
test_x = []
test_y = []
val_x = []
val_y = []
#optim.zero_grad()
total = 0
correct = 0
for prob in train_dataloader:
  outputs = net(prob).tolist()
  train_x += outputs
  train_y += prob.is_sat
for prob in test_dataloader:
  outputs = net(prob).tolist()
  test_x += outputs
  test_y += prob.is_sat
for prob in val_dataloader:
  outputs = net(prob).tolist()
  val_x += outputs
  val_y += prob.is_sat
test_accuracy_lr = evaluate(train_x, train_y, test_x, test_y, val_x, val_y)

#if not print_screen:
#print("logistic regression: test accuracy for linear readout: " + str(test_accuracy_lr), file=log_file, flush=True)
#print("svm: test accuracy for linear readout: " + str(test_accuracy_svm), file=log_file, flush=True)
#else:
print("logistic regression: test accuracy for linear readout: " + str(test_accuracy_lr))
#print("svm: test accuracy for linear readout: " + str(test_accuracy_svm))

    

