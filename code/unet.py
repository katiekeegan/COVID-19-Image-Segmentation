import os
import numpy as np
import pathlib
import sys
import keras
import datetime 
import json
import collections
import tqdm
import imageio
import scipy
import operator
import pandas as pd
import cv2
from keras import backend as K
import skimage
import csbdeep
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
import sys
os.chdir('../SegGradCAM')
from seggradcam.dataloaders import Cityscapes
#from seggradcam.metrics import IoU, Dice
from seggradcam.unet import csbd_unet, manual_unet, TrainUnet
from seggradcam.training_write import TrainingParameters, TrainingResults
from seggradcam.training_plots import plot_predict_and_gt, plot_loss, plot_metric
from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
import matplotlib.pyplot as plt
from seggradcam.visualize_sgc import SegGradCAMplot

import joblib
#loading the data
x_train = joblib.load('../data/x_train.joblib')
y_train = joblib.load('../data/y_train.joblib')
x_val = joblib.load('../data/x_val.joblib')
y_val = joblib.load('../data/y_val.joblib')

x_train_new = x_train.reshape(-1,128,128,1)
y_train_new = y_train.reshape(-1,128,128,1)
x_val_new = x_val.reshape(-1,128,128,1)
y_val_new = y_val.reshape(-1,128,128,1)

BATCH_SIZE=16
N_TRAIN = 200
N_VAL = 50
trainparam = TrainingParameters(
    dataset_name = 'TexturedMnist',
                 n_classes=2
                ,scale = 1
                ,batch_size = BATCH_SIZE
                ,last_activation = 'sigmoid'
                ,n_depth = 4
                ,n_filter_base = 32  # 16
                ,pool = 2
                ,lr = 1.e-4
                ,epochs = 500 #00 #                        CHANGE THE N OF EPOCHS
                ,validation_steps = int(N_VAL/BATCH_SIZE)
                ,steps_per_epoch = int(N_TRAIN/BATCH_SIZE)
                ,loss = "binary_crossentropy"
                #,optimizer = Adam(lr=3.e-4)
                ,metrics = ['accuracy','IoU','Dice']
                ,input_shape=(128, 128,1)
                ,n_train= N_TRAIN
                ,n_val = N_VAL
)
trainparam.saveToJson()

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
from keras.utils import to_categorical
y_train_new_new = to_categorical(y_train_new)
y_val_new_new = to_categorical(y_val_new)

trainunet = TrainUnet(trainparam)
trainunet.csbdUnet()
fit_out = trainunet.fit_generator(datagen.flow(x_train_new,y_train_new_new),datagen.flow(x_val_new,y_val_new_new))

trainunet.model.save('../models/unet_model')
