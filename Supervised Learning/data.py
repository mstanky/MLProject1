import numpy as np
from sklearn.model_selection import train_test_split


class DiabetesData(object):
    def __init__(self, data_file):
        # read data into numpy array
        self.data = np.loadtxt(open(data_file, "rb"), delimiter=",", skiprows=1)
        # get column headers
        with open(data_file, 'r') as f:
            self.data_headers = f.readline().strip().split(',')[:-1]
        # separate out into features and labels, and split .75/.25 training and test data
        features = self.data[:, :-1]
        labels = np.array([self.data[:, -1]]).T
        self.X = features
        self.Y = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels)

class HousingData(object):
    def __init__(self, data_file):
        # read data into numpy array
        self.data = np.loadtxt(open(data_file, "rb"), delimiter=",", skiprows=1)
        # get column headers
        with open(data_file, 'r') as f:
            self.data_headers = f.readline().strip().split(',')[:-1]
        # separate out into features and labels, and split .75/.25 training and test data
        features = self.data[:, 1:]
        labels = np.array([self.data[:, 0]]).T
        self.X = features
        self.Y = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels)

if __name__ == "__main__":
    housing_data = HousingData("Housing.csv")
    diabetes_data = DiabetesData("diabetes.csv")
    print(np.shape(housing_data.Y))