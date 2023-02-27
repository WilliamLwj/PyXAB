# -*- coding: utf-8 -*-
"""
Real-life Data
============================
In this example, we run the VHCT algorithm to tune hyperparameters for Support Vector Machine (SVM)
"""

from PyXAB.algos.VHCT import VHCT
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import plot_regret

# Useful functions and classes for the learning process
from sklearn import svm
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle


# %%
# We first define the objective as maximizing the ROC_AUC score on the testing dataset after training the SVM using
# the hyperparameters with the training dataset.

class obj_func_landmine():

    def __init__(self, X_train, Y_train, X_test, Y_test):

        self.X_train = X_train              # Training X
        self.Y_train = Y_train              # Training Y
        self.X_test = X_test                # Testing X
        self.Y_test = Y_test                # Testing Y
        self.fmax = 1

    def f(self, point):
        C = point[0]                        # First parameter
        gam = point[1]                      # Second parameter

        clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)       # The machine learning model is SVM
        clf.fit(self.X_train, self.Y_train)                                 # Fit the model using training data
        pred = clf.predict_proba(self.X_test)                               # Make prediction on the testing
        score = roc_auc_score(self.Y_test, pred[:, 1])                      # The reward is the ROC_AUC score

        return score

# %%
# Input the data and then split the data into the training dataset and the testing dataset. Then define the objective
# function using the datasets.

landmine_data = pickle.load(open("../../PyXAB/landmine/landmine_formated_data.pkl", "rb"))
all_X_train, all_Y_train, all_X_test, all_Y_test = landmine_data["all_X_train"], landmine_data["all_Y_train"], \
                                                       landmine_data["all_X_test"], landmine_data["all_Y_test"]

X_train = all_X_train[0]
Y_train = np.squeeze(all_Y_train[0])
X_test = all_X_test[0]
Y_test = np.squeeze(all_Y_test[0])

target = obj_func_landmine(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

# %%
# Define the number of rounds, the domain, the partition, and the algorithm for the learning process
T = 500
domain = [[1e-4, 10.0], [1e-2, 10.0]]
partition = BinaryPartition
algo = VHCT(domain=domain, rho=0.5, partition=partition)

# To plot the regret, we can initialize the cumulative regret and the cumulative regret list

cumulative_regret = 0
cumulative_regret_list = []


# %%
# In each iteration of the learning process, the algorithm calls the ``pull(t)`` function to obtain a point, and then
# the reward for the point is returned to the algorithm by calling ``receive_reward(t, reward)``.

for t in range(1, T+1):

    point = algo.pull(t)
    reward = target.f(point)
    algo.receive_reward(t, reward)
    inst_regret = target.fmax - target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)


# plot the regret
plot_regret(np.array(cumulative_regret_list), name='VHCT')
