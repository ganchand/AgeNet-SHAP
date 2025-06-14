import pandas as pd
import numpy as np
import os

def data_processing(csv_file):
    data = pd.read_csv(csv_file)   # csv_file should be in format of ROI1, ROI2,...ROIlast, Age, Subject ID
    
    # normalize the ROIs
    data['Sum'] = data.iloc[:, :-2].sum(axis=1)
    numeric_columns = data.columns[:-3]
    data[numeric_columns] = data[numeric_columns].div(data['Sum'], axis=0)

    X = data.drop(['Subject ID', 'Age','Sum'], axis =1 ).values
    y =  data['Age'].values
    y = np.reshape(y,(-1,1))
    
    Predictors = list((data.drop(['Subject ID', 'Age','Sum'], axis =1)).columns)
    input_shape = data.drop(['Subject ID', 'Age','Sum'], axis =1 ).values.shape[1]
    
    return X, y, Predictors, input_shape
    