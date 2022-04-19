import numpy as np
import matplotlib.pyplot as plt
import imageio 
from tools import *
def converttobinary_individual_image(y_pred):
    for k in range(0,len(y_pred)):
        if y_pred[k] >= 0.5:
            y_pred[k] = 1
        else:
            y_pred[k] = 0
    return y_pred 

def meshgrid(image):
  ylist = np.linspace(0, image.shape[0], image.shape[0])
  xlist = np.linspace(0, image.shape[1], image.shape[1])
  return np.meshgrid(xlist, ylist)

def see_predictions_mask_rf(image, rf_model):
    predictions = rf_model.predict(image)
    predictions_binary = converttobinary_individual_image(predictions)
    predictions_binary_reshaped = predictions_binary.reshape(128,128)
    return predictions_binary_reshaped

def see_predictions_mask_svr(image, svr_model):
    predictions = svr_model.predict(image)
    predictions_binary = converttobinary_individual_image(predictions)
    predictions_binary_reshaped = predictions_binary.reshape(128,128)
    return predictions_binary_reshaped

def see_predictions_mask_unet(image, model):
    predictions = model.predict(np.expand_dims(image, 0))[0]
    predictions_binary = converttobinary_individual_image(predictions)
    return predictions_binary

def see_predictions_overlay(predictions_binary_mask, image, save ='false', filename = ''): #mask needs to be 2D (128,128)
    ximg = np.expand_dims(image,axis=2)
    preds = predictions_binary_mask
    
    X, Y = meshgrid(np.expand_dims(image,axis=2))
    plt.imshow(ximg, vmin=0, vmax=1,cmap='gray')
    plt.axis('off')
    plt.contour(X, Y, preds, colors='pink')
    if save == 'true':
        plt.savefig(filename)
        
