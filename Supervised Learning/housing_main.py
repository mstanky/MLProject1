from sklearn import tree
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC, SVR
from data import DiabetesData, HousingData

DATA_FILE = 'Housing.csv'
RUN_ALL = False
RUN_LC = True
RUN_VC = True

if __name__ == "__main__":
    np.random.seed(1000)
    data = HousingData(DATA_FILE)
    # train and score trees
    if RUN_ALL or RUN_LC:
        proportions = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        _, axes = plt.subplots(1, 5, figsize=(30, 5))
        # learning_curves
        sizes, train_score, test_score = learning_curve(tree.DecisionTreeRegressor(), data.X, data.Y,
                                                        train_sizes=proportions,
                                                        cv=5, shuffle=True,
                                                        scoring='r2')
        avg_train, avg_test = np.mean(train_score, axis=1), np.mean(test_score, axis=1)
        axes[0].set_title("Decision Tree")
        axes[0].plot(proportions, avg_test, 'o:')
        axes[0].plot(proportions, avg_train, 'o-')
        axes[0].set_xlabel("Training Samples")
        axes[0].set_ylabel("R2")

        sizes, train_score, test_score = learning_curve(
            MLPRegressor(max_iter=10000), data.X, data.Y.T[0], train_sizes=proportions, cv=5,
            scoring='r2')
        avg_train, avg_test = np.mean(train_score, axis=1), np.mean(test_score, axis=1)
        axes[1].set_title("Neural Network")
        axes[1].plot(proportions, avg_test, 'o:')
        axes[1].plot(proportions, avg_train, 'o-')
        axes[1].set_xlabel("Training Samples")
        axes[1].set_ylabel("R2")

        sizes, train_score, test_score = learning_curve(
            GradientBoostingRegressor(), data.X, data.Y.T[0],
            train_sizes=proportions, cv=5, scoring='r2'
        )
        avg_train, avg_test = np.mean(train_score, axis=1), np.mean(test_score, axis=1)
        axes[2].set_title("Boosted Classifier")
        axes[2].plot(proportions, avg_test, 'o:')
        axes[2].plot(proportions, avg_train, 'o-')
        axes[2].set_xlabel("Training Samples")
        axes[2].set_ylabel("R2")

        sizes, train_score, test_score = learning_curve(
            KNeighborsRegressor(), data.X, data.Y.T[0],
            train_sizes=proportions, cv=5, scoring='r2'
        )
        avg_train, avg_test = np.mean(train_score, axis=1), np.mean(test_score, axis=1)
        axes[3].set_title("KNearestNeighbors")
        axes[3].plot(proportions, avg_test, 'o:')
        axes[3].plot(proportions, avg_train, 'o-')
        axes[3].set_xlabel("Training Samples")
        axes[3].set_ylabel("R2")

        scaler = StandardScaler().fit_transform(data.X)
        sizes, train_score, test_score = learning_curve(
            LinearSVC(max_iter=10000), scaler, data.Y.T[0],
            train_sizes=proportions, cv=5, scoring='r2'
        )
        avg_train, avg_test = np.mean(train_score, axis=1), np.mean(test_score, axis=1)
        axes[4].set_title("Linear SVM")
        axes[4].plot(proportions, avg_test, 'o:')
        axes[4].plot(proportions, avg_train, 'o-')
        axes[4].set_xlabel("Training Samples")
        axes[4].set_ylabel("R2")

        plt.savefig('Housing - Learning Curves - Samples.png')
        plt.clf()

        # learning curves - iterations
        _, axes = plt.subplots(1, 2, figsize=(15, 5))
        iters = [(i + 1) * 200 for i in range(10)]
        train_scores, test_scores = [], []
        for i in iters:
            sizes, train_score, test_score = learning_curve(
                LinearSVC(max_iter=i), scaler, data.Y.T[0], train_sizes=[.4],
                cv=5, scoring='r2'
            )
            train_scores.append(np.mean(train_score))
            test_scores.append(np.mean(test_score))
        axes[0].set_title("Linear SVM - Max Training Iterations")
        axes[0].plot(iters, test_scores, 'o:')
        axes[0].plot(iters, train_scores, 'o-')
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("R2")

        iters = [(i + 1) * 2000 for i in range(10)]
        train_scores, test_scores = [], []
        for i in iters:
            sizes, train_score, test_score = learning_curve(
                MLPRegressor(max_iter=i), data.X, data.Y.T[0], train_sizes=[.5],
                cv=5, scoring='r2'
            )
            train_scores.append(np.mean(train_score))
            test_scores.append(np.mean(test_score))
        axes[1].set_title("Neural Network - Max Training Iterations")
        axes[1].plot(iters, test_scores, 'o:')
        axes[1].plot(iters, train_scores, 'o-')
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("R2")

        plt.savefig('Housing - Learning Curves - Iterations.png')
        plt.clf()
        plt.show()

    if RUN_ALL or RUN_VC:
        _, axes = plt.subplots(5, 2, figsize=(20, 20))
        # DT - pruning (max depth) and min samples per leaf
        depth = [i + 1 for i in range(10)]
        fit_time, score_time, test_score, train_score = np.zeros(10), np.zeros(10), \
                                                        np.zeros(10), np.zeros(10)
        for i in depth:
            dt = tree.DecisionTreeRegressor(max_depth=i)
            results = cross_validate(dt, data.X, data.Y, scoring='r2', return_train_score=True)
            fit_time[i - 1] = np.mean(results['fit_time'])
            score_time[i - 1] = np.mean(results['score_time'])
            test_score[i - 1] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[0][0].set_title("DT Validation Curve - Depth")
        axes[0][0].plot(depth, test_score, 'o:')
        axes[0][0].plot(depth, train_score, 'o-')

        min_samples = [max(1, (i * 10)) for i in range(10)]
        for i in range(len(min_samples)):
            dt = tree.DecisionTreeRegressor(min_samples_leaf=min_samples[i])
            results = cross_validate(dt, data.X, data.Y, scoring='r2', return_train_score=True)
            fit_time[i] = np.mean(results['fit_time'])
            score_time[i] = np.mean(results['score_time'])
            test_score[i] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[0][1].set_title("DT Validation Curve - Min Samples/Leaf")
        axes[0][1].plot(min_samples, test_score, 'o:')
        axes[0][1].plot(min_samples, train_score, 'o-')
        # NN - hidden layer size & Alpha
        hidden_layer_sizes = [i + 1 for i in range(10)]
        fit_time, score_time, test_score = np.zeros(10), np.zeros(10), np.zeros(10)
        for i in hidden_layer_sizes:
            nn = MLPRegressor(hidden_layer_sizes=(i), max_iter=5000)
            results = cross_validate(nn, data.X, data.Y.T[0], scoring='r2', return_train_score=True)
            fit_time[i - 1] = np.mean(results['fit_time'])
            score_time[i - 1] = np.mean(results['score_time'])
            test_score[i - 1] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[1][0].set_title("NN Validation Curve - Hidden Layer Size")
        axes[1][0].plot(hidden_layer_sizes, test_score, 'o:')
        axes[1][0].plot(hidden_layer_sizes, train_score, 'o-')

        alphas = [.0001 * (i + 1) for i in range(10)]
        for i in range(len(alphas)):
            nn = MLPRegressor(hidden_layer_sizes=9, max_iter=5000, alpha=i)
            results = cross_validate(nn, data.X, data.Y.T[0], scoring='r2', return_train_score=True)
            fit_time[i - 1] = np.mean(results['fit_time'])
            score_time[i - 1] = np.mean(results['score_time'])
            test_score[i - 1] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[1][1].set_title("NN Validation Curve - Alpha")
        axes[1][1].plot(alphas, test_score, 'o:')
        axes[1][1].plot(alphas, train_score, 'o-')

        # SVM - kernel type and C Parameter
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        scaler = StandardScaler().fit_transform(data.X)
        for kernel in kernels:
            learner = SVR(kernel=kernel)
            results = cross_validate(learner, scaler, data.Y.T[0], scoring='r2', return_train_score=True)
            print('SVM Performance - {} Kernel'.format(kernel))
            print("Testing")
            print(np.mean(results['test_score']))
            print("Training")
            print(np.mean(results['train_score']))

        c_vals = [(i + 1) / 10 for i in range(10)]
        for c in range(len(c_vals)):
            learner = SVR(kernel='sigmoid', C=c_vals[c])
            results = cross_validate(learner, scaler, data.Y.T[0], scoring='r2', return_train_score=True)
            fit_time[c] = np.mean(results['fit_time'])
            score_time[c] = np.mean(results['score_time'])
            test_score[c] = np.mean(results['test_score'])
            train_score[c] = np.mean(results['train_score'])
        axes[2][1].set_title("SVM Validation Curve - Regularization Parameter (C)")
        axes[2][1].plot(c_vals, test_score, 'o:')
        axes[2][1].plot(c_vals, train_score, 'o-')
        # KNN - K and leaf size
        k = [i + 1 for i in range(10)]
        for i in k:
            knn = KNeighborsRegressor(n_neighbors=i, )
            results = cross_validate(knn, data.X, data.Y.T[0], scoring='r2', return_train_score=True)
            fit_time[i - 1] = np.mean(results['fit_time'])
            score_time[i - 1] = np.mean(results['score_time'])
            test_score[i - 1] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[3][0].set_title("KNN Validation Curve - K")
        axes[3][0].plot(k, test_score, 'o:')
        axes[3][0].plot(k, train_score, 'o-')

        leaf_sizes = [(i+1)*5 for i in range(10)]
        weights = ['uniform', 'distance']
        for i in range(len(weights)):
            knn = KNeighborsRegressor(n_neighbors=3, weights=weights[i])
            results = cross_validate(knn, data.X, data.Y.T[0], scoring='r2', return_train_score=True)
            # fit_time[i - 1] = np.mean(results['fit_time'])
            # score_time[i - 1] = np.mean(results['score_time'])
            # test_score[i - 1] = np.mean(results['test_score'])
            # train_score[i - 1] = np.mean(results['train_score'])
            print(weights[i])
            print("training")
            print(np.mean(results['train_score']))
            print("testing")
            print(np.mean(results['test_score']))
        # axes[3][1].set_title("KNN Validation Curve - Leaf Size")
        # axes[3][1].plot(leaf_sizes, test_score, 'o-')
        # axes[3][1].plot(leaf_sizes, train_score, 'o-')

        # Boosted Learner - num learners & Weights
        num_learners = [1000 * (i + 1) for i in range(10)]
        for i in range(len(num_learners)):
            boosted = AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=4),
                                                 n_estimators=num_learners[i])
            results = cross_validate(boosted, data.X, data.Y.T[0], scoring='r2', return_train_score=True)
            fit_time[i - 1] = np.mean(results['fit_time'])
            score_time[i - 1] = np.mean(results['score_time'])
            test_score[i - 1] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[4][0].set_title("Boosted Classifier Validation Curve - Num Learners")
        axes[4][0].plot(num_learners, test_score, 'o:')
        axes[4][0].plot(num_learners, train_score, 'o-')

        learning_rates = [(i+1)/10 for i in range(10)]
        for i in range(len(learning_rates)):
            boosted = AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=4),
                                         learning_rate=learning_rates[i],
                                         n_estimators=800)
            results = cross_validate(boosted, data.X, data.Y.T[0], scoring='r2', return_train_score=True)
            fit_time[i - 1] = np.mean(results['fit_time'])
            score_time[i - 1] = np.mean(results['score_time'])
            test_score[i - 1] = np.mean(results['test_score'])
            train_score[i - 1] = np.mean(results['train_score'])
        axes[4][1].set_title("Boosted Classifier Validation Curve - Learning Rate")
        axes[4][1].plot(learning_rates, test_score, 'o:')
        axes[4][1].plot(learning_rates, train_score, 'o-')
        plt.savefig('Housing - Validation Curves.png')