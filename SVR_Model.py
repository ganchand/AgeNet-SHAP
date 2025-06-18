import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn import metrics
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

from DataProcessing import data_processing

data_file = 'path to data CSV file'   # should be in the format ROI1, ROI2, ROI3, ..., ROIlast, Age, Subject ID

X, y, Predictor, input_shape = data_processing(data_file)

epsilon_range = np.arange(-7,1)
c_range = np.arange(-1,4,1)
gamma_range = np.arange(-12,-2,1)

gamma_grid = []
for i in range(1,len(gamma_range)):
    sqr=2**int(gamma_range[i])
    gamma_grid.append(sqr)

epsilon_grid = []
for i in range(0 ,len(epsilon_range)):
    sqr= 2**int(epsilon_range[i])
    epsilon_grid.append(sqr)
c_grid=[2]

print(gamma_grid)
print(epsilon_grid)
print(c_grid)

param_grid = {'C': c_grid,
              'gamma': gamma_grid,
              'epsilon': epsilon_grid}

print(param_grid)

cv = KFold(n_splits=10, random_state=45, shuffle=True)

for i, (train_index, test_index) in enumerate(cv.split(X)):
    model = SVR(kernel='rbf')
    print(metrics.SCORERS.keys())
    grid = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', refit=True)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_result = grid.fit(X_train, np.ravel(y_train))
    
    best_parameters = grid_result.best_params_
    best_score = grid_result.best_score_
    best_svr = SVR(kernel='rbf', C=best_parameters["C"], gamma=best_parameters["gamma"],
                  epsilon=best_parameters['epsilon'])

    clf=best_svr.fit(X_train, np.ravel(y_train))
    
    print('Grid best parameter (lowest error): ', best_parameters)
    print('Grid best score (lowest error): ', best_score)
    fold_scores = []
    original_score = metrics.mean_absolute_error(np.ravel(y_test), clf.predict(X_test))
    
    Predictions = clf.predict(X_test)
    train_predictions = clf.predict(X_train)
    train_score = metrics.mean_absolute_error(np.ravel(y_train), train_predictions)
    test_score = metrics.mean_absolute_error(np.ravel(y_test), Predictions)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print('Test score', test_score)
    print('Train score', train_score)
