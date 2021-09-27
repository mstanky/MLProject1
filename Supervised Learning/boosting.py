from sklearn.metrics import recall_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier


class BoostedClf(object):
    def __init__(self):
        self.clf = GradientBoostingClassifier()
        self.report = None
        self.recall = None

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def score(self, x_test, y_test):
        y_pred = self.clf.predict(x_test)
        self.recall = recall_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred)
