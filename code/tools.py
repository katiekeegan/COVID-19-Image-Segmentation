import sklearn 
import pandas as pd
import mdai
import os
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib import patches
from skimage import color
from skimage import io 
from numpy import savetxt
from sklearn.metrics import mean_squared_error
import random
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from skimage.color import gray2rgb, rgb2gray, label2rgb
import pickle
from PIL import Image
import cv2

#function from md.ai, modified slightly for the annotation display that I want
def display_annotations(image,boxes,masks,class_ids,scores=None,title="",figsize=(8,8),ax=None,show_mask=True,show_bbox=True):
    # Number of instancesload_mask
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True
#    # Generate random colors
#    colors = colors or mdai.visualize.random_colors(N)
        
#    colors = []
#    for i in range(N):
#        colors.append(color)
    color = (1,1,1)
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    #ax.set_title(title)
    #masked_image = image.astype(np.uint32).copy()
    masked_image = np.zeros(image.shape)
    for i in range(N):
        #color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)
            
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = mdai.visualize.apply_mask(masked_image, mask, color)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    plt.close()
    return masked_image.astype(np.uint8)

def l2_relative_error(y_preds,y_val):
  errors = []
  for i in range(0,len(y_preds)):
    numerator = np.linalg.norm(y_preds[i]-y_val[i])
    denominator = np.linalg.norm(y_val[i])
    errors.append(numerator/denominator)
  return np.mean(errors)

def converttobinary(y_preds): #takes the y_preds results as input
  y_preds = np.array(y_preds)
  binary_y_preds = np.empty((y_preds.shape))
  for i in range(0,y_preds.shape[0]):
    y_pred = y_preds[i,:]
    binary_y_pred = converttobinary_individual_image(y_pred)
    binary_y_preds[i,:] = binary_y_pred
  return binary_y_preds

def converttobinary_individual_image(y_pred):
    for k in range(0,len(y_pred)):
        if y_pred[k] >= 0.5:
            y_pred[k] = 1
        else:
            y_pred[k] = 0
    return y_pred 

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def average_dice(y_preds, y_val):
    if y_preds.shape != y_val.shape:
        raise ValueError("Shape mismatch: y_preds and y_val must have the same shape.")

    dice_coeffs = []
    num_samples = y_preds.shape[0]
    for sample in range(num_samples):
        dice_coeff = dice(y_preds[sample], y_val[sample])
        dice_coeffs.append(dice_coeff)
        
    return np.mean(dice_coeffs)
