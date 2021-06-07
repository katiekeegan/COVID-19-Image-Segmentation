#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:03:22 2021

@author: katie
"""

from pathlib import Path
import pydicom
import pandas as pd
import mdai
import os
from shutil import copy2
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

#paths to data
json_path = '/Users/katie/Documents/GitHub/COVID19ImageProject/MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
images_path = Path('/Users/katie/Documents/GitHub/COVID19ImageProject/manifest-1608266677008')
filenames = list(images_path.glob('**/*.dcm'))
info = []
PATH_TO_IMAGES = '/Users/katie/Documents/GitHub/COVID19ImageProject/reformatted'

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
    color = (0.22, 1, 0.078)
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


#dataset preprocessing
p = mdai.preprocess.Project(
                annotations_fp=json_path,
                images_dir=PATH_TO_IMAGES
            )
labels_dict = {
        'L_neeyDn' : 1
        }
p.set_labels_dict(labels_dict)
dataset = p.get_dataset_by_id('D_q24k8z')
dataset.prepare()
images = []
images_with_anns = []
color = (0.22, 1, 0.078)
list_of_errors = []
print(len(dataset.get_image_ids()))
for i in range(0,len(dataset.get_image_ids())):
    try: 
        print(i)
        image_id = dataset.get_image_ids()[i]
        image, class_ids, bboxes, masks = mdai.visualize.get_image_ground_truth(image_id, dataset)
        images.append(image)
        image_with_anns = display_annotations(image,bboxes,masks,class_ids,show_mask=True,show_bbox=False)
        images_with_anns.append(image_with_anns)
    except UnboundLocalError:
        list_of_errors.append(i)
        
#vectorizing the images (turning the images and their corresponding annotations into 1-dimensional arrays to feed into model)
for i in range(0,len(images)):
    print(i)
    img = images[i]
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(-1)
    img_stacked_max = img_stacked.max()
    img_stacked = (1/img_stacked_max) * img_stacked
    images[i] = img_stacked
for i in range(0,len(images_with_anns)):
    print(i)
    img_w_anns = images_with_anns[i]
    img_w_anns_rows, img_w_anns_columns = img_w_anns.shape[:2] 
    img_w_anns_stacked = img_w_anns.reshape(-1)
    img_w_anns_stacked_max = img_w_anns_stacked.max()
    img_w_anns_stacked = (1/img_w_anns_stacked_max) * img_w_anns_stacked
    images_with_anns[i] = img_w_anns_stacked
    
#splitting the data
x_train, x_val, y_train, y_val = train_test_split(images, images_with_anns, test_size=0.2, random_state=0)

#setting random forest refressor
rf = RandomForestRegressor()

#setting parameters for GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [10,100,500,1000],
    'max_features': [2, 5,10],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10,50,100, 200,500,700,1000]
}

#GridSearchCV will go through each possible combination of the parameters in param_grid
model = GridSearchCV(estimator = rf, param_grid = param_grid,verbose=3,scoring='neg_mean_squared_error',cv=3, n_jobs=1)

model_result = model.fit(x_train, y_train)
best_params = model_result.best_params_
print(best_params)