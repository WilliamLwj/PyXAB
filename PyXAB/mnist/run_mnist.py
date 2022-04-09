import pdb
import pickle

import numpy as np

from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.algos import *
from PyXAB.mnist.mnist_obj import obj_func_mnist
from tensorboardX import SummaryWriter


import os

import argparse

parser = argparse.ArgumentParser(description="MNIST Hyperparmeter Tuning")

parser.add_argument('--rounds', type=int, default=500, help="number of rounds.")

parser.add_argument('--ALGO', default='T-HOO', type=str,
                    help='Algorithm to be used')

parser.add_argument('--save_path', default='logs', type=str, metavar='PATH',
                    help='path to save train log, test log and ckpt (default: logs)')



def run_MNIST(algo, rounds, tb_logger):

    # batch_size, lr, weight_decay
    domain = [[1, 500], [1e-6, 1], [1e-6, 5e-2]]
    partition = BinaryPartition
    target = obj_func_mnist()
    algo_dictionary = {'T-HOO': HOO.T_HOO(rounds=rounds, rho=0.5, domain=domain, partition=partition),
                       'HCT': HCT.HCT(domain=domain, rho=0.5, partition=partition),
                       'VHCT': VHCT.VHCT(domain=domain, rho=0.5, partition=partition),
                       'POO': POO.POO(domain=domain, partition=partition, algo=HOO.T_HOO),
                       'PCT': PCT.PCT(domain=domain, partition=partition)}

    algorithm = algo_dictionary[algo]
    print(algo, ": training")

    regret = 0
    for t in range(1, rounds + 1):
        print(t)
        point = algo.pull(t)
        train_loss, train_acc, val_loss, val_acc = target.f(point)
        algo.receive_reward(t, val_acc)
        inst_regret = target.fmax - target.f(point)
        regret += inst_regret
        if tb_logger is not None:
            tb_logger.add_scalar('Train/Loss', train_loss, t)
            tb_logger.add_scalar('Train/Acc', train_acc, t)
            tb_logger.add_scalar('Test/Loss', val_loss, t)
            tb_logger.add_scalar('Test/Acc', val_acc, t)
            tb_logger.add_scalar('Regret', regret, t)


if __name__ == "__main__":

    args = parser.parse_args()

    tb_logger = SummaryWriter(args.save_path + '/log')
    if not os.path.exists(args.save_path + '/log'):
        print('Create {}.'.format(args.save_path + '/log'))
        os.makedirs(args.save_path + '/log')

    num_clients = args.M
    T = args.rounds
    run_MNIST(args.ALGO, T, tb_logger)
