#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mdai
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pydicom
import SimpleITK as sitk
import nibabel
from mayavi import mlab
import skimage

JSON = './MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
results = mdai.common_utils.json_to_dataframe(JSON)
annots_df = results['annotations']
labels_df = results['labels']

metadata = './metadata.csv'
metadata_df = pd.read_csv(metadata)

#function for convertinga row of annotation data to a binary mask, from MD.ai documentation
def load_mask_instance(row):
    """Load instance masks for the given annotation row. Masks can be different types,
    mask is a binary true/false map of the same size as the image.
    """
    
    mask = np.zeros((512, 512), dtype=np.uint8)

    annotation_mode = row.annotationMode
    # print(annotation_mode)

    if annotation_mode == "bbox":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        w = int(row["data"]["width"])
        h = int(row["data"]["height"])
        mask_instance = mask[:,:].copy()
        cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
        mask[:,:] = mask_instance

    # FreeForm or Polygon
    elif annotation_mode == "freeform" or annotation_mode == "polygon":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:,:].copy()
        cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
        mask[:,:] = mask_instance

    # Line
    elif annotation_mode == "line":
        vertices = np.array(row["data"]["vertices"])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:,:].copy()
        cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
        mask[:,:] = mask_instance

    elif annotation_mode == "location":
        # Bounding Box
        x = int(row["data"]["x"])
        y = int(row["data"]["y"])
        mask_instance = mask[:,:].copy()
        cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
        mask[:,:] = mask_instance

    elif annotation_mode is None:
        print("Not a local instance")
        
    return mask

def remove_dup(a):
   i = 0
   while i < len(a):
      j = i + 1
      while j < len(a):
         if a[i] == a[j]:
            del a[j]
         else:
            j += 1
      i += 1
   return(a)
   
def append_SeriesUID(df): #metadata
    SeriesUID_items = []
    for i in range(0,len(df)):
        SeriesUID_items.append(df.iloc[i]['Series UID'])
    reduced_entries = remove_dup(SeriesUID_items)
    reduced_entries_new = [x for x in reduced_entries if str(x) != 'nan']
    print(len(reduced_entries_new))
    return(reduced_entries_new)
   
#According to the metadata spreadsheet, there appears to be a 
#correspondence between the SeriesUID for the DICOM files
#and the SeriesInstanceUID. This function constructs binary masks 
#from the rows of annotation data (each of which has a SeriesInstanceUID)
#and saves them in the corresponding folder in the order that they appeared in the 
#original annotations file.
def create_masks():
    reduced_SeriesUID_entries = append_SeriesUID(metadata_df)
#    for k in range(0,len(reduced_StudyUID_entries)):
    for k in range(0,len(reduced_SeriesUID_entries)):
        a = []
        for i in range(0,len(annots_df)):
            if reduced_SeriesUID_entries[k] == annots_df.iloc[i].SeriesInstanceUID:
                a.append(i)
        #arrays=[]
        for m in range(0,len(a)):
            b = load_mask_instance(annots_df.iloc[m])
            #arrays.append(b)
        #volume = np.stack(arrays,axis=2)
        #print(volume.shape)
            plt.imsave('./annotations{}/image_{:0>3}.png'.format((metadata_df.iloc[k]['File Location']).replace('.','',1),m),b)
     