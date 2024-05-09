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
if args.model_file != "":
    net.load_state_dict(torch.load(args.model_file))

time = strftime("%Y-%m-%d %H:%M:%S", localtime()).replace(" ", "_")
task_name = args.task_name + '_' + time + '_lr' + str(args.learning_rate) + '_label' + str(args.label_proportion)

log_file = open(os.path.join(args.log_dir, task_name + '.log'), 'a+')

for arg in vars(args):
    if args.screen:
        print(arg, getattr(args, arg))
    else:
        print(arg, getattr(args, arg), flush=True, file=log_file)

sigmoid = nn.Sigmoid()

print_screen = args.screen

best_acc = 0.0
start_epoch = 0


def print_grad(model):
    grads = []
    for name, param in model.named_parameters():
        if param.grad != None:
            grads.append(param.grad.view(-1))
        # grads.append(param.grad.view(-1))
        # print(name, param.grad)
        # print(name)
        # print(torch.count_nonzero(param.grad) / torch.numel(param.grad))
        # print(" ")
        # grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    # print("non zero " + str(torch.count_nonzero(grads) / len(grads)))
    print(torch.sort(torch.absolute(grads)))
    # print(len(grads))


if not args.debug:
    if args.wandb_proj is None:
        args.wandb_proj = "supervised"

    wandb.login(key=args.wandb_login)
    wandb.init(project=args.wandb_proj, entity=args.wandb_entity, name=task_name)
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

# labels = labels[:5000]
# n_vars = n_vars[:5000]
print(len(labels))
train_size = int(len(labels) * args.label_proportion)
if not args.debug:
    wandb.config.train_size = train_size
print("size of training set: " + str(train_size))
test_size = min(len(labels) - train_size, 2000)

train_labels = labels[test_size:test_size + train_size]
train_n_vars = n_vars[test_size:test_size + train_size]
test_labels = labels[:test_size]
test_n_vars = n_vars[:test_size]

torch.manual_seed(0)
val_dataset = SATInstances(args, start=0, data_length=test_size, labels=test_labels, n_vars=test_n_vars, augment=False)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn_test, shuffle=False)
# val_dataset = generate_train(args, problem_len=500)
# val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,collate_fn=collate_fn_test, num_workers=args.num_workers)
torch.set_printoptions(profile="full")
train_dataset = None
train_dataloader = None
if args.label_proportion != 0:
    train_dataset = SATInstances(args, start=test_size, data_length=train_size, labels=train_labels,
                                 n_vars=train_n_vars, augment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  collate_fn=collate_fn_test, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

supervised_train = supervised(args.dim)
supervised_train = supervised_train.cuda()
optim = optim.Adam(list(net.parameters()) + list(supervised_train.parameters()), lr=args.learning_rate,
                   weight_decay=args.weight_decay)
best_acc = 0
# train1 = generate(args)
for epoch in range(start_epoch, args.epochs):

    # val_bar = tqdm(val)

    if epoch % args.test_epoch == 0:
        net.eval()
        supervised_train.eval()

        optim.zero_grad()
        total = 0
        correct = 0
        for prob in val_dataloader:
            outputs = net(prob)
            # print(outputs[:2])
            target = torch.Tensor(prob.is_sat).cuda().long()
            # for i in range(len(outputs)):
            #    for j in range(i+1, len(outputs)):
            #        if torch.equal(outputs[i], outputs[j]):
            #            print(i, j)
            # print("bug")
            # print(target)
            # print(" ")
            total_cur, correct_cur = supervised_train.evaluate(outputs, target)
            total += total_cur
            correct += correct_cur
        accuracy = correct / total
        # print(total)
        if not args.debug:
            print("test accuracy: " + str(accuracy), file=log_file, flush=True)
        else:
            print("test accuracy: " + str(accuracy))

        if not args.debug:
            wandb.log({"validation accuracy": accuracy})

        if accuracy > best_acc:
            best_acc = accuracy
            if args.save_model:
                print("update model")
                torch.save(net.state_dict(), os.path.join(args.model_dir, "net_" + str(jobid) + ".pth"))
                torch.save(supervised_train.state_dict(),
                           os.path.join(args.model_dir, "supervised_train_" + str(jobid) + ".pth"))

        if args.train_accuracy:
            total = 0
            correct = 0
            for prob in train_dataloader:
                outputs = net(prob)
                # print(outputs)
                target = torch.Tensor(prob.is_sat).cuda().long()
                total_cur, correct_cur = supervised_train.evaluate(outputs, target)
                total += total_cur
                correct += correct_cur
            accuracy = correct / total
            # print(total)
            if not args.debug:
                print("train accuracy: " + str(accuracy), file=log_file, flush=True)
            else:
                print("train accuracy: " + str(accuracy))
            if not args.debug:
                wandb.log({"train accuracy": accuracy})

    # print(epoch)
    if args.label_proportion == 0:
        train_dataset = generate_train(args)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=collate_fn_test, num_workers=args.num_workers)

    if not args.debug:
        print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc), file=log_file, flush=True)
    else:
        print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc))
    # print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), flush=True)

    net.train()
    supervised_train.train()
    train_loss = 0
    batch = 0
    # with autograd.detect_anomaly():
    for (_, prob) in enumerate(train_dataloader):
        optim.zero_grad()
        outputs = net(prob)
        # print(outputs)
        # for i in range(len(outputs)):
        #      for j in range(i+1, len(outputs)):
        #          print(torch.norm(outputs[i]-outputs[j]))
        # if torch.equal(outputs[i], outputs[j]):
        #    print(i, j)
        # print(outputs)
        # if _ == 0:
        #    print(prob.is_sat)
        # print(outputs)
        # print_grad(net)
        target = torch.Tensor(prob.is_sat).cuda().long()
        loss = supervised_train.loss(outputs, target)
        train_loss += loss
        batch += 1
        desc = 'loss: %.4f; ' % (loss.item())
        if not args.debug:
            wandb.log({"train loss": loss})
        loss.backward()
        # print_grad(net)
        optim.step()
    print(train_loss / batch)
    # if _ % 50 == 0:
    # wandb.log({"train loss": loss})
    # if not args.debug:
    # print(desc, flush=True, file = log_file)
    # else:
    # print(desc)
    # print(desc, flush=True, file = log_file)
