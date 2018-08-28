


import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import cross_validation
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score
import utilities


# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)


params = {'kernel': ['rbf'], 'gamma': [0.05, 0.01, 0.005, 0.001],
'C': [1, 10, 50, 100, 150, 300, 600]}



def classifier(C, gamma):
    val = cross_val_score(
        SVC(C=C, gamma=gamma, random_state=5),
        X_train, y_train, 'recall_weighted', cv=5
    ).mean()
    return val




gp_params = {"alpha": 1e-5}
clfBO = BayesianOptimization(classifier, {'C': (1, 600), 'gamma': (0.001, 0.01)})




clfBO.explore({'C': [10, 150, 10, 300, 400], 'gamma': [0.001, 0.01, 0.001, 0.01, 0.01]})
clfBO.maximize(n_iter=10, **gp_params)
print('-' * 53)



print('#' * 53)
print('Final Results')
print('SVC: %f' % clfBO.res['max']['max_val'])



params = {'kernel': 'rbf', 'gamma' : 0.0100, 'C' : 574.777}
classifier = SVC(**params)
classifier.fit(X_train, y_train) 
y_true, y_pred = y_test, classifier.predict(X_test)
print "\nFull performance report:\n"
print classification_report(y_true, y_pred)

