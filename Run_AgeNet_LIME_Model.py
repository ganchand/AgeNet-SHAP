import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from pandas import ExcelWriter    
import tensorflow as tf
from keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop, Adagrad,Adadelta, RMSprop,Adam
%matplotlib inline
from sklearn.model_selection import StratifiedKFold 
import shap
import lime
import lime.lime_tabular
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from AgeNet_Model import build_model
from DataProcessing import data_processing

seed = 100     # random seed for reproducibility
np.random.seed(seed)
tf.random.set_seed(seed)

data_file = 'path to data CSV file'   # should be in the format ROI1, ROI2, ROI3, ..., ROIlast, Age, Subject ID

X, y, Predictor, input_shape = data_processing(data_file)

# Define callbacks
def lr_scheduler(epoch, lr):
    decay_rate = 0.9
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.00001, verbose=1, mode='auto', min_delta=0.01)
early_call_train = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=50, verbose=2, mode='min', restore_best_weights=True)

n_splits = 10
batch_size = 64
n_epoch = 300
rate = 0.01

# Set up KFold cross-validation
cv = KFold(n_splits=n_splits, random_state=123, shuffle=True)

# Start cross-validation training-testing loop
for i, (train_index, test_index) in enumerate(cv.split(X)):
    model = build_model(input_shape)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=rate), loss=tf.keras.losses.MeanAbsoluteError(), metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

    fold_number = i + 1
    print('fold_number', fold_number)
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=2,
                               validation_split=0.2, callbacks=[reduce_lr, early_call_train])

    # Make predictions
    Predictions = model.predict(X_test)  # save to get the final correlation between actual and predicted age
    score1_test = model.evaluate(X_test, y_test, verbose=0)
    score1_train = model.evaluate(X_train, y_train, verbose=0)
    print('Test score', score1_test)
    print('Train score', score1_train)
    
    # Run LIME
    def model_predict(X):
            return model.predict(X)
        
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=Predictors, 
                        discretize_continuous=False,feature_selection='lasso_path',mode='regression',random_state=i)
    
    for l in range(len(X_test)):
        instance = X_test[l]
        exp = explainer_lime.explain_instance(instance, model_predict, num_features=22, num_samples=2000,
                                     model_regressor=ElasticNet(alpha=0.05, l1_ratio=0.5))
        
        feature_values = pd.DataFrame([x[1] for x in exp.as_list()], columns=['feature_values'])
        feature_range = pd.DataFrame([x[0] for x in exp.as_list()], columns=['feature_range'])
        row = pd.concat([feature_range,feature_values], axis=1)
        row['Test_index'] = test_index[l]
        row['fold_number'] = fold_number
        row['Percentage change'] = sheets
        row['Test_ids'] = sub_ids[test_index[l]]
        row['abs_feature_values'] = row['feature_values'].abs()
        print(row)
