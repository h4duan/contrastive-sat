import argparse
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp

from neurosat import NeuroSAT
from neurosatsimp import NeuroSATSimp

from projections import proj_v1 as proj

from data_maker import generate_train
from torch.utils.data import DataLoader

from sat_dataset import SATInstances
from sat_dataset import SATInstancesAugmentTwo
from sat_dataset import collate_fn_test
from sat_dataset import collate_fn_train

from config import parser
from loss_gpu import ssl
import random

import time
from time import gmtime, strftime
import numpy as np

import logging

from pprint import pformat

import wandb


class SSLModelWrapper(nn.Module):
    def __init__(self, emb_f, proj_f):
        super(SSLModelWrapper, self).__init__()
        self.emb_f = emb_f
        self.proj_f = proj_f

    def forward(self, x1, x2):
        z1 = self.emb_f(x1)
        z2 = self.emb_f(x2)

        z1 = self.proj_f(z1)
        z2 = self.proj_f(z2)

        return z1, z2


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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


def train_loop(rank, args):
    if args.world_size == 1:
        rank = torch.cuda.current_device()

    logger_role = (args.world_size == 1) or rank == 0  # If this rank is the one in charge of logging (i.e., rank 0)
    logger = get_logger(args.log_dir, args.task_name)

    logging.info(f"Process {rank}/{args.world_size} started.")
    if logger_role:
        logger.info(pformat(args))

    setup(rank, args.world_size)

    if args.neurosat:
        net = NeuroSAT(args)
    else:
        net = NeuroSATSimp(args)
    project = proj(in_dim=args.dim)
    model = SSLModelWrapper(net, project).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    ssl_train = ssl()

    optimizer = optim.Adam(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_acc_lr = 0.0
    start_epoch = 0

    if args.val_folder is None:
        logger.error("No validation set")
        quit()

    logger.info("Loading validation set")

    label_file = os.path.join(args.val_folder, args.val_label)
    with open(label_file, 'rb') as handle:
        labels = pickle.load(handle)
    n_vars_file = os.path.join(args.val_folder, args.val_n_vars)
    with open(n_vars_file, 'rb') as handle:
        n_vars = pickle.load(handle)

    labels = labels[:10000]
    n_vars = n_vars[:10000]

    train_size = int(len(labels) * args.label_proportion)
    val_size = int((len(labels) - train_size) * 0.5)
    test_size = len(labels) - train_size - val_size

    test_labels = labels[:test_size]
    test_n_vars = n_vars[:test_size]
    val_end = test_size + val_size
    val_labels = labels[test_size:val_end]
    val_n_vars = n_vars[test_size:val_end]
    train_labels = labels[val_end:]
    train_n_vars = n_vars[val_end:]

    if not args.debug and logger_role:
        logger.info("Setting up WandB")
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
        wandb.init(project=args.wandb_proj, entity=args.wandb_entity, name=task_name)
        wandb.config.update(args)
        wandb.config.train_size = train_size
        wandb.watch(ddp_model, log_freq=100)
        wandb.config.jobid = os.environ["SLURM_JOB_ID"]

    if logger_role:
        logger.info("Loading datasets...")
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
    if logger_role:
        logger.info("Datasets loaded.")
    if args.simclr and args.vicreg:
        logger.error("You can only choose only ssl loss!")
        quit()

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch} started on rank {rank}.")
        if epoch % args.test_epoch == 0 and logger_role and not args.debug:
            ddp_model.eval()

            train_x = []
            train_y = []
            test_x = []
            test_y = []
            val_x = []
            val_y = []
            optimizer.zero_grad()

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

            logger.info(f"Evaluating model on rank {rank}...")
            test_accuracy_lr = ssl_train.evaluate(train_x, train_y, test_x, test_y, val_x, val_y)
            if not args.debug:
                wandb.log({"test_accuracy_lr": test_accuracy_lr})
            logger.info(f"Evaluation done! Logistic regression test accuracy for linear readout: {test_accuracy_lr}")

            if test_accuracy_lr > best_acc_lr:
                best_acc_lr = test_accuracy_lr
                if args.save_model:
                    logger.info(f"Saving model to {args.model_path}")
                    torch.save(net.state_dict(), args.model_path)
                    logger.info(f"Saving model DONE!")
                if not args.debug:
                    wandb.log({"best_test_accuracy_lr": best_acc_lr})

            if test_accuracy_lr < best_acc_lr - 0.2 and epoch > 500:
                logger.info("Training finished!")
                quit()

            if args.train_file is None:
                logger.info("Generating data online")

        logger.info(f"Augment dataset")
        augmentation_dataset = generate_train(args)
        augmentation_dataloader = DataLoader(augmentation_dataset, shuffle=False, batch_size=args.batch_size,
                                             collate_fn=collate_fn_train, num_workers=args.num_workers)

        logger.info(f"Data Augmentation Done!")
        if logger_role:
            logger.info(f"==> Epoch {epoch + 1}/{args.epochs}, previous best lr: {best_acc_lr:.3f}")

        ddp_model.train()
        t_loss = 0
        batch = 0

        for _, (prob1, prob2) in enumerate(augmentation_dataloader):
            optimizer.zero_grad()
            if logger_role:
                logger.info(f"ddp_model(prob1, prob2)")
            outputs1, outputs2 = ddp_model(prob1, prob2)

            if logger_role:
                logger.info(f"all_gather()...")
            with torch.no_grad():
                all_outs1 = [torch.zeros_like(outputs1) for _ in range(args.world_size)]
                all_outs2 = [torch.zeros_like(outputs2) for _ in range(args.world_size)]

                torch.distributed.all_gather(all_outs1, outputs1)
                torch.distributed.all_gather(all_outs2, outputs2)

            if logger_role:
                logger.info(f"all_gather() RELEASE!")

            all_outs1[rank] = outputs1
            all_outs2[rank] = outputs2

            all_outs1 = torch.cat(all_outs1)
            all_outs2 = torch.cat(all_outs2)

            all_outs1 = F.normalize(all_outs1, dim=1)
            all_outs2 = F.normalize(all_outs2, dim=1)

            loss = None
            if args.simclr:
                loss = ssl_train.simclr_loss(all_outs1, all_outs2, tau=args.simclr_tau)
            elif args.vicreg:
                loss = ssl_train.vicreg_loss(all_outs1, all_outs2, lamb=args.vicreg_lambda, mu=args.vicreg_mu,
                                             nu=args.vicreg_nu)
            elif args.siamese:
                loss = ssl_train.siamese_loss(all_outs1, all_outs2)
            if logger_role:
                logger.info('Batch loss: %.4f; ' % (loss.item()))

            loss.backward()
            logger.info(f"{rank}: Backward done!")
            t_loss += loss.item()
            batch += 1

            optimizer.step()
            logger.info(f"{rank}: Step done!")

        if logger_role:
            logger.info("Batch done!")
        ave_loss = t_loss / batch
        if not args.debug and logger_role:
            wandb.log({"ssl_loss": ave_loss})
        if logger_role:
            logger.info(f"Avg loss {ave_loss}")

    logger.info("Training finished!")

    cleanup()


def get_logger(log_dir, log_file):
    logger = logging.getLogger('train')
    logFormatter = logging.Formatter("%(asctime)s  [" + os.environ["SLURM_JOB_ID"] + "] [%(levelname)-5.5s]  %(message)s")
    logger.setLevel(logging.INFO)

    # Handlers.
    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(logFormatter)

    fileHandler = logging.FileHandler(os.path.join(log_dir, log_file + '.log'))
    fileHandler.setFormatter(logFormatter)

    # logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    return logger


if __name__ == '__main__':
    args = parser.parse_args()

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "_")
    task_name = f'{args.task_name}_{start_time}_lp{args.label_proportion}'
    args.task_name = task_name
    args.model_path = os.path.join(args.model_dir,
                              f"neurosat_ssl_{args.world_size}xgpu_sr10to40" + start_time + ".pth")

    mp.spawn(train_loop,
             args=(args,),
             nprocs=args.world_size,
             join=True)
