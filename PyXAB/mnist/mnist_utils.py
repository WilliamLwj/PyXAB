
import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
from math import pow, pi, cos

def MNIST_dataset(path='/scratch/gilbreth/li3549/SVRGdata'):
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST(path, download=True, train=False, transform=transform)
    train_set = datasets.MNIST(path, download=True, train=True, transform=transform)
    return train_set, test_set


def MNIST_one_layer():
    # Create the nn model
    input_size = 784
    hidden_sizes = [64]
    output_size = 10

    # The non-convex function (model) is Sequential function fron torch. Notice that we have implemented an activation function to make it non-convex
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], output_size),
        nn.LogSoftmax(dim=1))

    return model



def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))


class AverageCalculator():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        assert (n > 0)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)
