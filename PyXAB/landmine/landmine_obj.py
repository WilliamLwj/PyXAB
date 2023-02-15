from ..synthetic_obj.Objective import Objective

from sklearn import svm
from sklearn.metrics import roc_auc_score


class obj_func_landmine(Objective):
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
