import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class linear_class(nn.Module):
    def __init__(self, in_dim):
        self.fc = nn.Linear(in_dim, 2)

    def loss(self, x, y):
        loss = nn.CrossEntropyLoss()
        return loss(self.fc(x), y)

class ssl(nn.Module):
    # def __init__(self, in_dim):
    #     super(ssl, self).__init__()
    #     self.in_dim = in_dim
    #     hidden_dim = in_dim
    #     self.fc1 = nn.Linear(in_dim, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, in_dim)
    #     self.bn1 = nn.BatchNorm1d(hidden_dim)
    #     self.bn2 = nn.BatchNorm1d(in_dim)
    #     hidden_dim = int(in_dim / 4)
    #     self.fc3 = nn.Linear(in_dim, hidden_dim)
    #     self.fc4 = nn.Linear(hidden_dim, in_dim)
    #     self.bn3 = nn.BatchNorm1d(hidden_dim)
        #self.tau = tau

    def simclr_loss(self, z1, z2, tau):
        #project
        # z1 = self.fc2(F.relu(self.fc1(z1)))
        # z2 = self.fc2(F.relu(self.fc1(z2)))
        # z1 = F.normalize(z1, dim=1)
        # # print(torch.norm(g1[0]))
        # z2 = F.normalize(z2, dim=1)
        sim_score_1 = torch.matmul(z1, torch.transpose(z2, 0, 1)) / tau
        sim_score_2 = torch.matmul(z2, torch.transpose(z1, 0, 1)) / tau
        sim_score_self_1 = torch.matmul(z1, torch.transpose(z1, 0, 1)) / tau
        sim_score_self_2 = torch.matmul(z2, torch.transpose(z2, 0, 1)) / tau
        # sim_score_self_1.fill_diagonal_(0)
        # sim_score_self_2.fill_diagonal_(0)
        # print(sim_score_self_1)
        sim_score_self_1 = sim_score_self_1[~torch.eye(sim_score_self_1.shape[0], dtype=bool)].reshape(
            sim_score_self_1.shape[0], -1)
        sim_score_self_2 = sim_score_self_2[~torch.eye(sim_score_self_2.shape[0], dtype=bool)].reshape(
            sim_score_self_2.shape[0], -1)
        sim_score_1 = torch.cat((sim_score_1, sim_score_self_1), 1)
        sim_score_2 = torch.cat((sim_score_2, sim_score_self_2), 1)
        # print(torch.sum(torch.diagonal(sim_score_1, 0)))
        # print(torch.sum(torch.diagonal(sim_score_2, 0)))
        # print(sim_score_self_1)
        loss_1 = -torch.sum(torch.diagonal(sim_score_1, 0)) + torch.sum(torch.logsumexp(sim_score_1, dim=1))
        loss_2 = -torch.sum(torch.diagonal(sim_score_2, 0)) + torch.sum(torch.logsumexp(sim_score_2, dim=1))
        return (loss_1 + loss_2) / (2 * len(z1))


    def siamese_loss(self, z1, z2):
        z1 = self.bn2(self.fc2(F.relu(self.bn1(self.fc1(z1)))))
        z2 = self.bn2(self.fc2(F.relu(self.bn1(self.fc1(z2)))))
        def D(p, z):
            z_clone = z.detach()
            p_norm = F.normalize(p, dim=1)
            #print(p)
            #print(z)
            #print(" ")
            z_norm = F.normalize(z_clone, dim=1)
            #print(p_norm, z_norm)
            return -(p_norm * z_norm).sum(dim=1).mean()
        p1 = self.fc4(F.relu(self.bn3(self.fc3(z1))))
        p2 = self.fc4(F.relu(self.bn3(self.fc3(z2))))

        return (D(p1, z2) + D(p2, z1)) / 2

    def vicreg_loss(self, z1, z2, lamb, mu, nu):
        z1 = self.fc2(F.relu(self.fc1(z1)))
        z2 = self.fc2(F.relu(self.fc1(z2)))

        #invariance_loss
        mse_loss = nn.MSELoss()
        invariance_loss = mse_loss(z1, z2)

        #variance loss
        epsilon = 1e-6
        gamma = 1
        std_z1 = torch.sqrt(torch.var(z1, dim = 0, unbiased = True) + epsilon)
        std_z2 = torch.sqrt(torch.var(z2, dim = 0, unbiased = True) + epsilon)
        variance_loss = torch.mean(F.relu(gamma - std_z1))
        variance_loss += torch.mean(F.relu(gamma - std_z2))

        #covariance loss
        N = len(z1)
        D = len(z1[0])
        norm_z1 = z1 - torch.mean(z1, dim = 0)
        norm_z2 = z2 - torch.mean(z2, dim = 0)
        p_cov_z1 = torch.square(torch.matmul(norm_z1, torch.transpose(norm_z1, 0, 1)) / (N-1))
        p_cov_z2 = torch.square(torch.matmul(norm_z2, torch.transpose(norm_z2, 0, 1)) / (N-1))
        covariance_loss = (torch.sum(p_cov_z1) - torch.sum(torch.diagonal(p_cov_z1, 0))) / D
        covariance_loss += (torch.sum(p_cov_z2) - torch.sum(torch.diagonal(p_cov_z2, 0)) /D)

        return lamb * invariance_loss + mu * variance_loss + nu * covariance_loss




    def evaluate(self, train_x, train_y, test_x, test_y, val_x, val_y):
        #possible_c = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        possible_c = [100000, 0.001, 1000, 100, 10, 1, 0.1, 0.01, 10000]
        best_val_lr = 0
        best_val_svm = 0
        test_lr = 0
        test_svm = 0
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y, dtype = int)
        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y, dtype = int)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y, dtype = int)
        #best_c = 0
        for c in possible_c:
            clf_lr = LogisticRegression(max_iter = 20000, C = c).fit(train_x, train_y)
            #clf_svm = SVC(gamma="auto", C = c).fit(train_x, train_y)
            local_lr = clf_lr.score(val_x, val_y)
            #local_svm = clf_svm.score(val_x, val_y)
            if local_lr > best_val_lr:
                #print(local_lr, best_val_lr)
                best_val_lr = local_lr
                test_lr = clf_lr.score(test_x, test_y)
                #print(local_lr, test_lr)
            """
            if local_svm > best_val_svm:
                #print(local_svm, best_val_svm)
                best_val_svm = local_svm
                test_svm = clf_svm.score(test_x, test_y)
                #best_c = c
                #print(local_svm, test_svm)
            """
        #print(best_c)
        return test_lr



class supervised(nn.Module):
    def __init__(self, in_dim):
        super(supervised, self).__init__()
        self.fc = nn.Linear(in_dim, 2)

    def loss(self, z, y):
        loss = nn.CrossEntropyLoss()
        return loss(self.fc(z), y)

    def evaluate(self, z, y):
        logits = self.fc(z)
        pred = torch.argmax(logits, dim = 1)
        return len(y), (pred == y).sum()


