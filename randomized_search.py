


import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.model_selection import RandomizedSearchCV
import utilities



# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)



parameter_grid = {'kernel': ['rbf'], 'gamma': [0.05, .01, .005, 0.001], 'C': [1, 10, 50, 100, 150, 300, 600]}



metrics = ['precision', 'recall_weighted']



for metric in metrics:
    print "\n#### Searching optimal hyperparameters for", metric

    classifier = RandomizedSearchCV(SVC(), parameter_grid, cv=5, random_state=10,                     scoring=metric, n_iter=3)
    classifier.fit(X_train, y_train)

    print "\nScores across the parameter grid:"
    for params, avg_score, _ in classifier.grid_scores_:
        print params, '-->', round(avg_score, 3)

    print "\nHighest scoring parameter set:", classifier.best_params_
    y_true, y_pred = y_test, classifier.predict(X_test)
    print "\nFull performance report:\n"
    print classification_report(y_true, y_pred)

