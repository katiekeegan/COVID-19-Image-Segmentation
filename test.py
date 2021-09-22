from pathlib import Path
import pydicom
import pandas as pd
import mdai
from shutil import copy2
import sklearn 
from PIL import Image
import cv2
import random
from skimage.color import gray2rgb, rgb2gray, label2rgb
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib import patches
from sklearn.model_selection import train_test_split
import joblib


#from code.tools import display_annotations

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

def directory_preprocessing():
    images_path = Path('../manifest-1612365584013')
    filenames = list(images_path.glob('**/*.dcm'))
    info = []
    overall_dir = '../images'
    for i in range(0, len(StudyInstanceUID_reduced)):
        StudyInstanceUID_dir = '{}'.format(StudyInstanceUID_reduced[i])
        StudyInstanceUID_path = os.path.join(overall_dir, StudyInstanceUID_dir)
        os.mkdir(StudyInstanceUID_path)
        SeriesInstanceUID_items = []
        for k in range(0,len(df)):
            if StudyInstanceUID_reduced[i] == df.iloc[k].StudyInstanceUID:
                SeriesInstanceUID_items.append(df.iloc[k].SeriesInstanceUID)
        SeriesInstanceUID_reduced = remove_dup(SeriesInstanceUID_items)
        for j in range(0,len(SeriesInstanceUID_reduced)):
            SeriesInstanceUID_dir = '{}'.format(SeriesInstanceUID_reduced[j])
            SeriesInstanceUID_path = os.path.join(StudyInstanceUID_path, SeriesInstanceUID_dir)
            os.mkdir(SeriesInstanceUID_path)
            #print(SeriesInstanceUID_path)
            for b in range(0,len(df)):
                if SeriesInstanceUID_reduced[j] == df.iloc[b].SeriesInstanceUID:
                    copy2(df.iloc[b].fn,'../images/{}/{}/{}.dcm'.format(StudyInstanceUID_reduced[i],SeriesInstanceUID_reduced[j],df.iloc[b].SOPInstanceUID))
                   
def see_label_groups():
    #dataset preprocessing
    PATH_TO_IMAGES = './data/images'
    json_path = './data/MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
    p = mdai.preprocess.Project(
                annotations_fp=json_path,
                images_dir=PATH_TO_IMAGES
            )
    p.show_labels()
    
def obtain_training_testing_data():
    #dataset preprocessing
    PATH_TO_IMAGES = './images'
    json_path = './MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
    p = mdai.preprocess.Project(
                annotations_fp=json_path,
                images_dir=PATH_TO_IMAGES
            )
    labels_dict = {
        'L_neeyDn' : 1 #infectious opacities in Purple Group 1 (group id: G_86Pmql)
        }
    p.set_labels_dict(labels_dict)
    dataset = p.get_dataset_by_id('D_q24k8z')
    dataset.prepare()
    images = []
    annotations = []
    color = (1, 1, 1)
    print(len(dataset.get_image_ids()))
    random.seed(a=42)
    randomlist = random.sample(range(0, len(dataset.get_image_ids())), 250)
    count = 0
    for i in randomlist:
        try: 
            print(i)
            image_id = dataset.get_image_ids()[i]
            
            #obtain image and masks
            image, class_ids, bboxes, masks = mdai.visualize.get_image_ground_truth(image_id, dataset)
            
            #the image may have three color channels, so this step ensures that image_gray has shape (512,512) and not (512,512,3)
            image_gray = rgb2gray(image)
            
            if image_gray.shape != (512,512):
                #ignore non-(512,512) images
                raise ValueError('This image does not have the correct shape.')
            
            else: 
                #include (512,512) image in the images list
                images.append(image_gray)
                
                #obtain annotations
                annotation = display_annotations(image,bboxes,masks,class_ids,show_mask=True,show_bbox=False)
                
                #convert annotation from (512,512,3) to (512,512)
                annotation_gray = rgb2gray(annotation)
                
                #unravel annotation_gray in order to go through it and convert to binary
                annotation_gray = annotation_gray.ravel()
                for k in range(0,len(annotation_gray)):
                    if annotation_gray[k] != 0:
                        annotation_gray[k] = 1
                        
                #reshape unraveled binary annotation_gray back to (512,512)
                annotation_gray = annotation_gray.reshape(512,512)
                
                #include (512,512) binary annotation_gray in annotations list
                annotations.append(annotation_gray)
                
                count += 1
                
        except UnboundLocalError or TypeError or ValueError:
                #list_of_errors.append(i)
                continue
            
        if count == 250:
            break
        
        else:
            continue
    
    images_downsampled = []
    annotations_downsampled = []
    for i in range(0,len(images)):
      images_downsampled.append(cv2.resize(images[i], dsize=(128,128), interpolation=cv2.INTER_CUBIC))
      annotations_downsampled.append(cv2.resize(annotations[i].reshape(512,512), dsize=(128,128), interpolation=cv2.INTER_CUBIC))

    images_new = np.empty((len(images),16384))
    for i in range(0,len(images)):
      images_new[i,:] = (images_downsampled[i].ravel()) / (images_downsampled[i].ravel()).max() #normalizing values to be between 0 and 1
    
    annotations_new = np.empty((len(annotations),16384))
    for i in range(0,len(annotations)):
      annotation = annotations_downsampled[i].ravel()
      for k in range(0,len(annotation)):
        if annotation[k] != 0:
          annotation[k] = 1
      annotations_new[i,:] = annotation
      
    x_train, x_val, y_train, y_val = train_test_split(images_new,annotations_new, test_size=0.2, random_state=50)
    joblib.dump(x_train, './data/x_train.joblib')
    joblib.dump(y_train, './data/y_train.joblib')
    joblib.dump(x_val, './data/x_val.joblib')
    joblib.dump(y_val, './data/y_val.joblib')
    return x_train, y_train, x_val, y_val