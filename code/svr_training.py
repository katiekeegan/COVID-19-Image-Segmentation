import pandas as pd
from sklearn.pipeline import Pipeline
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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skimage.color import gray2rgb, rgb2gray, label2rgb
import tools
import joblib
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.multioutput import MultiOutputRegressor

#loading the data
x_train = joblib.load('../data/x_train.joblib')
y_train = joblib.load('../data/y_train.joblib')
x_val = joblib.load('../data/x_val.joblib')
y_val = joblib.load('../data/y_val.joblib')

#initializing n_estimators and errors list
n_estimators_list = []
errors = []
errors_binary_list = []

#choosing c
c = 0.01
e  = 0
svr_model = MultiOutputRegressor(SVR(C=c, cache_size=50,verbose=4))

print(c)

#fit model
svr_model.fit(x_train,y_train)

y_preds = svr_model.predict(x_val)

joblib.dump(svr_model, '../results/svr_model_c_{}.joblib'.format(c)) 
joblib.dump(y_preds, '../results/svr_c_{}.joblib'.format(c))

y_preds_binary = tools.converttobinary(y_preds)

#save error for this model

errors_binary_list.append(tools.average_dice(y_preds_binary,y_val))
print("Binary Error: ")
print(errors_binary_list)
