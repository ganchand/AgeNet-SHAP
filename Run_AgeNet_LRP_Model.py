import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
from tensorflow.keras.models import load_model
import innvestigate
import innvestigate.utils as iutils
from pandas import ExcelWriter 
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from AgeNet_Model import build_model
from DataProcessing import data_processing

seed = 100     # random seed for reproducibility
np.random.seed(seed)
tf.random.set_seed(seed)

data_file = 'path to data CSV file'   # should be in the format ROI1, ROI2, ROI3, ..., ROIlast, Age, Subject ID

X, y, Predictor, input_shape = data_processing(data_file)

cv = KFold(n_splits=n_splits, random_state=123, shuffle=True)

for i, (train_index, test_index) in enumerate(cv.split(X)):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = load_model(f'path to the saved trained AgeNet-model.h5')
        
        Predictions = model.predict(X_test)
        score1_test = model.evaluate(X_test, y_test, verbose=0)
        score1_train = model.evaluate(X_train, y_train, verbose=0)

        # Run LRP
        analyzer = innvestigate.create_analyzer("lrp.epsilon", model, epsilon=1e-7 )
        relevance_scores = analyzer.analyze(X_test)
        print(relevance_scores)