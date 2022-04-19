import keras
import joblib
import tensorflow as tf
import numpy as np
#loading the data
x_train = joblib.load('../data/x_train.joblib')
y_train = joblib.load('../data/y_train.joblib')
x_val = joblib.load('../data/x_val.joblib')
y_val = joblib.load('../data/y_val.joblib')

x_train_new = x_train.reshape(-1,128,128,1)
y_train_new = y_train.reshape(-1,128,128,1)
x_val_new = x_val.reshape(-1,128,128,1)
y_val_new = y_val.reshape(-1,128,128,1)
trainunet = joblib.load('../models/trainunet.joblib')
model = trainunet.model
#model = keras.models.load_model('../models/unet_model.h5')
y_preds = np.empty((x_val_new.shape[0],16384))
for i in range(0,x_val_new.shape[0]):
    prediction = model.predict(np.expand_dims(x_val_new[1], 0))[0][:,:,0]
    prediction_unraveled = prediction.ravel()
    y_preds[i,:] = prediction_unraveled
    
joblib.dump(y_preds, '../results/y_preds_unet.joblib')
