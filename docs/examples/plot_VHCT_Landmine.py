# -*- coding: utf-8 -*-
"""
Real-life Data
============================
Run the VHCT algorithm to tune hyperparameters for Support Vector Machine (SVM)
"""

from PyXAB.algos import *
import matplotlib.pyplot as plt
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import plot_regret

from sklearn import svm
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle


class obj_func_landmine():

    def __init__(self, X_train, Y_train, X_test, Y_test):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.fmax = 1

    def f(self, point):
        C = point[0]
        gam = point[1]

        clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
        clf.fit(self.X_train, self.Y_train)
        pred = clf.predict_proba(self.X_test)
        score = roc_auc_score(self.Y_test, pred[:, 1])

        return score


landmine_data = pickle.load(open("../../PyXAB/landmine/landmine_formated_data.pkl", "rb"))
all_X_train, all_Y_train, all_X_test, all_Y_test = landmine_data["all_X_train"], landmine_data["all_Y_train"], \
                                                       landmine_data["all_X_test"], landmine_data["all_Y_test"]

X_train = all_X_train[0]
Y_train = np.squeeze(all_Y_train[0])
X_test = all_X_test[0]
Y_test = np.squeeze(all_Y_test[0])

target = obj_func_landmine(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)


domain = [[1e-4, 10.0], [1e-2, 10.0]]
partition = BinaryPartition
T = 500

algo = VHCT.VHCT(domain=domain, rho=0.5, partition=partition)
# regret and regret list
cumulative_regret = 0
cumulative_regret_list = []

for t in range(1, T+1):

    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = target.fmax - target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)


# plot the regret
plot_regret(np.array(cumulative_regret_list), name='VHCT')
