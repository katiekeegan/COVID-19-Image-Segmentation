import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib import patches
from skimage import color
from skimage import io 
from numpy import savetxt
from sklearn.metrics import mean_squared_error
import random
from sklearn.ensemble import RandomForestClassifier
from skimage.color import gray2rgb, rgb2gray, label2rgb
import tools

#loading the data
x_train = joblib.load('../data/x_train.joblib')
y_train = joblib.load('../data/y_train.joblib')
x_val = joblib.load('../data/x_val.joblib')
y_val = joblib.load('../data/y_val.joblib')

#initializing n_estimators and errors list
n_estimators_list = []
errors = []
errors_binary = []

#initializing rf_model
#rf_model = joblib.load('rf_model_24.joblib')

rf_model = RandomForestRegressor(n_estimators=0, n_jobs=-1, random_state=42, warm_start=True, verbose=4)

#iterating through trees, 24 at a time
for i in range(0,30):
    #increase number of trees by 24 (more efficient since # vCPU = 24)
    rf_model.n_estimators += 10

    print(rf_model.n_estimators)
    n_estimators_list.append(rf_model.n_estimators)

    #fit model
    rf_model.fit(x_train,y_train)

    y_preds = rf_model.predict(y_val)
    y_preds_binary = tools.converttobinary(y_preds)
    joblib.dump(y_preds, '../results/rf_{}_y_preds.joblib'.format(rf_model.n_estimators))

    #save error for this model
    errors_binary.append(tools.average_dice(y_preds_binary,y_val))
    print(i, errors_binary[i])
  
    joblib.dump(rf_model, '../results/rf_model_{}.joblib'.format(rf_model.n_estimators))
#joblib.dump(rf_model, 'rf_model_n_estimators_144.joblib')
#create dataframe from data
df = pd.DataFrame(columns=['Trees','Dice Coefficient'])
df['Trees'] = n_estimators_list
#df['Error'] = errors
df['Dice Coefficient'] = errors_binary

joblib.dump(df, '../results/rf_model_accuracy_results.joblib')
df.to_csv('../results/rf_training_results.csv')
