import argparse
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT
from neurodiver import NeuroDiver
from neurosatsimp import NeuroSATSimp

from data_maker import generate_train
from data_maker import generate_train_sgen
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

import wandb

from data_maker import generate_train_power_law
from data_maker import generate_train_double_power_law
from data_maker import generate_train_popularity_similarity

args = parser.parse_args()

net = None

if args.neurosat:
    # print("here")
    net = NeuroSAT(args)
else:
    net = NeuroSATSimp(args)

net = net.cuda()

time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "_")
task_name = args.task_name + '_' + time + '_lp' + str(args.label_proportion)

log_file = open(os.path.join(args.log_dir, task_name + '.log'), 'a+')

if not args.debug:
    jobid = os.environ["SLURM_JOB_ID"]

for arg in vars(args):
    if args.screen:
        print(arg, getattr(args, arg))
    else:
        print(arg, getattr(args, arg), flush=True, file=log_file)

print(" ")

sigmoid = nn.Sigmoid()

print_screen = args.debug

best_acc_lr = 0.0
start_epoch = 0


def print_grad(model):
    grads = []
    for name, param in model.named_parameters():
        # if torch.count_nonzero(param) != torch.numel(param):
        #    print(name)
        #    print(torch.numel(param) - torch.count_nonzero(param))
        # print(param.requires_grad)
        # print(name, param.grad)
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    print(torch.sort(torch.absolute(grads)))
    print(torch.norm(grads))


# if train is not None:
#  print('num of train batches: ', len(train), file = log_file, flush=True)

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

# labels = labels[:10000]
# n_vars = n_vars[:10000]

train_size = int(len(labels) * args.label_proportion)
val_size = min(int((len(labels) - train_size) * 0.5), 1000)
test_size = val_size

test_labels = labels[:test_size]
test_n_vars = n_vars[:test_size]
val_end = test_size + val_size
val_labels = labels[test_size:val_end]
val_n_vars = n_vars[test_size:val_end]
train_labels = labels[val_end:]
train_n_vars = n_vars[val_end:]

if not args.debug:
    wandb_proj = args.wandb_proj
    if wandb_proj is None:
        if args.data_source == "sr":
            wandb_proj = 'ssl_sr'
        elif args.data_source == "power":
            wandb_proj = 'ssl_power'
        elif args.data_source == "graph_coloring":
            wandb_proj = 'ssl_graph'
        else:
            wandb_proj = 'ssl'

    wandb.login(key=args.wandb_login)
    wandb.init(project=wandb_proj, entity=args.wandb_entity, name=task_name)
    wandb.config.update(args)
    wandb.config.train_size = train_size
    wandb.watch(net, log_freq=100)
    wandb.config.jobid = os.environ["SLURM_JOB_ID"]

test_dataset = SATInstances(args, start=0, data_length=test_size, labels=test_labels, n_vars=test_n_vars, augment=False)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn_test,
                             num_workers=args.num_workers)
val_dataset = SATInstances(args, start=test_size, data_length=val_size, labels=val_labels, n_vars=val_n_vars,
                           augment=False)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn_test,
                            num_workers=args.num_workers)
train_dataset = SATInstances(args, start=test_size + val_size, data_length=train_size, labels=train_labels,
                             n_vars=train_n_vars, augment=False)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn_test,
                              num_workers=args.num_workers)

augmentation_dataset = None

if args.data_source == "graph_coloring":
    augmentation_dataset = SATInstancesAugmentTwo(instances=None, args=args, n_vars=None, is_sat=None,
                                                  file_folder=args.augment_file, num_file=args.augment_num_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssl_train = ssl(in_dim=args.dim)
ssl_train = ssl_train.cuda()

if not args.debug:
    if os.path.isfile(os.path.join(args.model_dir, "net_" + str(jobid) + ".pth")):
        net.load_state_dict(torch.load(os.path.join(args.model_dir, "net_" + str(jobid) + ".pth")))
    if os.path.isfile(os.path.join(args.model_dir, "ssl_train_" + str(jobid) + ".pth")):
        ssl_train.load_state_dict(torch.load(os.path.join(args.model_dir, "ssl_train_" + str(jobid) + ".pth")))

optim = optim.Adam(list(net.parameters()) + list(ssl_train.parameters()), lr=args.learning_rate,
                   weight_decay=args.weight_decay)

if args.simclr and args.vicreg:
    print("ERROR: you can only choose only ssl loss")
    quit

total_added_clause = 0
ave_added_clause = 0
total_instance = 0

shuffle = True

test_epoch = args.test_epoch
if args.data_source == "graph_coloring":
    shuffle = False
    test_epoch = 1
for epoch in range(start_epoch, args.epochs):

    if epoch % args.test_epoch == 0:
        net.eval()
        ssl_train.eval()
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        val_x = []
        val_y = []
        optim.zero_grad()
        total = 0
        correct = 0
        for prob in train_dataloader:
            # print(outputs)
            outputs = net(prob).tolist()
            # print(outputs)
            train_x += outputs
            train_y += prob.is_sat
        for prob in test_dataloader:
            outputs = net(prob).tolist()
            # print(outputs)
            test_x += outputs
            test_y += prob.is_sat
        for prob in val_dataloader:
            outputs = net(prob).tolist()
            # print(outputs)
            val_x += outputs
            val_y += prob.is_sat
        # print(train_x)
        # print(train_y)
        test_accuracy_lr = ssl_train.evaluate(train_x, train_y, test_x, test_y, val_x, val_y)
        if not args.debug:
            wandb.log({"test_accuracy_lr": test_accuracy_lr})
            # wandb.log({"test_accuracy_svm": test_accuracy_svm})
        if not print_screen:
            print("logistic regression: test accuracy for linear readout: " + str(test_accuracy_lr), file=log_file,
                  flush=True)
            # print("svm: test accuracy for linear readout: " + str(test_accuracy_svm), file=log_file, flush=True)
        else:
            print("logistic regression: test accuracy for linear readout: " + str(test_accuracy_lr))
            # print("svm: test accuracy for linear readout: " + str(test_accuracy_svm))

        if test_accuracy_lr > best_acc_lr:
            best_acc_lr = test_accuracy_lr
            if args.save_model and not args.debug:
                torch.save(net.state_dict(), os.path.join(args.model_dir, "net_" + str(jobid) + ".pth"))
                torch.save(ssl_train.state_dict(), os.path.join(args.model_dir, "ssl_train_" + str(jobid) + ".pth"))
            if not args.debug:
                wandb.log({"best_test_accuracy_lr": best_acc_lr})

        if test_accuracy_lr < best_acc_lr - 0.2 and epoch > 500:
            print("training finishes", flush=True, file=log_file)
            quit

        if args.train_file is None:
            if print_screen:
                print('generate data online', flush=True)

    # for i in range(1000):
    # augmentation_dataset = None
    if args.data_source == "sr":
        augmentation_dataset = generate_train(args)
    if args.data_source == "sgen":
        augmentation_dataset = generate_train_sgen(args)
    if args.data_source == "power":
        augmentation_dataset = generate_train_power_law(args)
    if args.data_source == "double_power":
        augmentation_dataset = generate_train_double_power_law(args)
    if args.data_source == "popularity_similarity":
        augmentation_dataset = generate_train_popularity_similarity(args)

    # print(augmentation_dataset)
    augmentation_dataloader = DataLoader(augmentation_dataset, shuffle=shuffle, batch_size=args.batch_size,
                                         collate_fn=collate_fn_train, num_workers=args.num_workers)
    if not print_screen:
        print('==> %d/%d epoch, previous best lr : %.3f ' % (epoch + 1, args.epochs, best_acc_lr), file=log_file,
              flush=True)
    else:
        print('==> %d/%d epoch, previous best lr: %.3f ' % (epoch + 1, args.epochs, best_acc_lr))
    net.train()
    ssl_train.train()
    t_loss = 0
    batch = 0
    # print("new batch")
    for _, (prob1, prob2) in enumerate(augmentation_dataloader):
        optim.zero_grad()
        outputs1 = net(prob1)
        # print(outputs1)
        outputs2 = net(prob2)
        # print(outputs1[:2, :])
        # print(outputs2[:2, :])
        # print("here")
        if args.study_ve:
            # print("here")
            total_added_clause += prob1.added_clause
            total_instance += 2 * args.batch_size
            if args.debug:
                print("average added clauses " + str(total_added_clause / total_instance))
            else:
                wandb.log({"average added clauses": total_added_clause / total_instance})

        loss = None
        if args.simclr:
            loss = ssl_train.simclr_loss(outputs1, outputs2, tau=args.simclr_tau)
        elif args.vicreg:
            loss = ssl_train.vicreg_loss(outputs1, outputs2, lamb=args.vicreg_lambda, mu=args.vicreg_mu,
                                         nu=args.vicreg_nu)
        elif args.siamese:
            loss = ssl_train.siamese_loss(outputs1, outputs2)
        desc = 'loss: %.4f; ' % (loss.item())

        loss.backward()

        t_loss += loss.item()
        batch += 1
        # print_grad(net)
        optim.step()
        # if _ % 1 == 0:
        if not print_screen:
            print(desc, flush=True, file=log_file)
        else:
            print(desc)
        ave_loss = t_loss / batch
        if not args.debug:
            wandb.log({"ssl_loss": ave_loss})
        if batch > args.test_epoch:
            break
    # if not args.debug:
    #  wandb.log({"ssl_loss": ave_loss})
    if not print_screen:
        print("ave loss " + str(ave_loss), flush=True, file=log_file)
    else:
        print("ave loss " + str(ave_loss))

print("training finishes", flush=True, file=log_file)
# print("train size " + str(total))

# print(desc, flush=True, file = log_file)
