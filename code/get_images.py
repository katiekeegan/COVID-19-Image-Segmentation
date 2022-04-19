import imageio
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tools import *
from visualizing_tools import *
number = 1
best_performing_trees = 10


y_preds_sgd = joblib.load('../results/sgd_100_y_preds.joblib')
y_preds_rf = joblib.load('../results/rf_{}_y_preds.joblib'.format(best_performing_trees))
y_preds_unet = joblib.load('../results/y_preds_unet.joblib')
x_val = joblib.load('../data/x_val.joblib')
y_val = joblib.load('../data/y_val.joblib')

image = x_val[number].reshape(128,128)
binary_unet = y_preds_unet[number].reshape(128,128)
binary_rf = y_preds_rf[number].reshape(128,128)
binary_svm = y_preds_sgd[number].reshape(128,128)
ground_truth = y_val[number].reshape(128,128)


see_predictions_overlay(binary_unet, image, save='true',filename='../results/binary_unet_{}.png'.format(number))
see_predictions_overlay(binary_rf, image, save='true',filename='../results/binary_sgd_{}.png'.format(number))
see_predictions_overlay(binary_svm, image, save='true',filename='../results/binary_rf_{}.png'.format(number))
see_predictions_overlay(ground_truth, image, save='true', filename='../results/ground_truth_{}.png'.format(number))

unet = imageio.imread('../results/binary_unet_{}.png'.format(number))
svm = imageio.imread('../results/binary_sgd_{}.png'.format(number))
rf = imageio.imread('../results/binary_rf_{}.png'.format(number))

fig, axs = plt.subplots(4)
axs[0].imshow(unet)
axs[0].set_title('U-net')
axs[1].imshow(svm)
axs[1].set_title('SGD')
axs[2].imshow(rf)
axs[2].set_title('RF')
axs[3].imshow(rf)
axs[3].set_title('Ground Truth')
fig.savefig('../results/side_by_side_{}.png'.format(number))

data = []
types = types = ['SGD','RF','U-net']
fig = plt.figure(figsize = (5, 3))
plt.bar(types, data, color ='maroon', width = 0.4)
plt.savefig('../results/barplot.png')
