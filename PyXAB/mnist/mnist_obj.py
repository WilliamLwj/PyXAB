import math
import numpy as np
import pdb
from PyXAB.synthetic_obj.Objective import Objective
from PyXAB.mnist.mnist_utils import *
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader




class obj_func_mnist(Objective):

    def __init__(self):

        self.fmax = 1

    def train_epoch_SGD(self, model, optimizer, train_loader, loss_fn, flatten_img=True, device='cpu'):
        model.train()
        loss = AverageCalculator()
        acc = AverageCalculator()

        for images, labels in train_loader:
            images = images.to(device)
            if flatten_img:
                images = images.view(images.shape[0], -1)
            yhat = model(images)
            labels = labels.to(device)
            loss_iter = loss_fn(yhat, labels)

            # optimization
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()

            # logging
            acc_iter = accuracy(yhat, labels)
            loss.update(loss_iter.data.item())
            acc.update(acc_iter)

        return loss.avg, acc.avg

    def validate_epoch(self, model, val_loader, loss_fn, flatten_img=True, device='cpu'):
        """One epoch of validation
        """
        model.eval()
        loss = AverageCalculator()
        acc = AverageCalculator()

        for images, labels in val_loader:
            images = images.to(device)
            if flatten_img:
                images = images.view(images.shape[0], -1)
            yhat = model(images)
            labels = labels.to(device)

            # logging
            loss_iter = loss_fn(yhat, labels)
            acc_iter = accuracy(yhat, labels)
            loss.update(loss_iter.data.item())
            acc.update(acc_iter)

        return loss.avg, acc.avg

    def f(self, param):

        batch = int(param[0])
        lr = param[1]  # learning rate
        weight_decay = param[2]

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print("Using device: {}".format(device))

        train_set, val_set = MNIST_dataset()
        flatten_img = True

        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=True)

        model = MNIST_one_layer().to(device)

        n_epoch = 60  # the number of epochs
        loss_fn = nn.NLLLoss()  # The loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(n_epoch):
            train_loss, train_acc = self.train_epoch_SGD(model, optimizer, train_loader, loss_fn,
                                                    flatten_img=flatten_img, device=device)

            val_loss, val_acc = self.validate_epoch(model, val_loader, loss_fn, flatten_img=flatten_img, device=device)


        return train_loss, train_acc, val_loss, val_acc
