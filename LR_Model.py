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

param_grid = {'alpha': (0.15, 0.2, 0.25, 0.3, 0.35)}
print(param_grid)

cv = KFold(n_splits=10, random_state=45, shuffle=True)

for i, (train_index, test_index) in enumerate(cv.split(X)):
    model = Lasso()
    grid = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', refit=True)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_result = grid.fit(X_train, np.ravel(y_train))
    
    best_parameters = grid_result.best_params_
    best_score = grid_result.best_score_
    best_lasso = Lasso(alpha=best_parameters["alpha"],max_iter=100000)
    clf=best_lasso.fit(X_train, np.ravel(y_train))
    
    print('Grid best parameter (lowest error): ', best_parameters)
    print('Grid best score (lowest error): ', best_score)
    fold_scores = []
    original_score = metrics.mean_absolute_error(np.ravel(y_test), clf.predict(X_test))
    
    Predictions = clf.predict(X_test)
    train_predictions = clf.predict(X_train)
    train_score = metrics.mean_absolute_error(np.ravel(y_train), train_predictions)
    test_score = metrics.mean_absolute_error(np.ravel(y_test), Predictions)
    print('Test score', test_score)
    print('Train score', train_score)
