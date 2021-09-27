import numpy
from sklearn import tree
import numpy as np
from sklearn.metrics import recall_score, classification_report


class DecisionTrees(object):
    def __init__(self):
        self.trees = np.array([None for i in range(100)])
        self.scores = np.array([None for i in range(100)])
        self.best_tree = None
        self.best_tree_report = None
        self.best_tree_recall = None

    def fit_trees(self, x_train, y_train):
        for i in range(len(self.trees)):
            dt = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=i+1)
            dt.fit(x_train, y_train)
            self.trees[i] = dt

    def score_trees(self, x_test, y_test):
        for i in range(len(self.scores)):
            y_pred = self.trees[i].predict(x_test)
            score = recall_score(y_test, y_pred)
            self.scores[i] = score
        ix = np.argmax(self.scores)
        self.best_tree = self.trees[ix]
        y_pred = self.trees[ix].predict(x_test)
        self.best_tree_report = classification_report(y_test, y_pred)
        self.best_tree_recall = self.scores[ix]
