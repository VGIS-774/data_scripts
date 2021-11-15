#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from PIL import Image


# In[3]:


videos_path = "dataset/Videos"
segmentations_path = "dataset/InstanceSegmentation"


# In[4]:


if not os.path.exists('mini_project_dataset/person_in_water'):
    os.makedirs('mini_project_dataset/person_in_water')
    
if not os.path.exists('mini_project_dataset/no_person_in_water'):
    os.makedirs('mini_project_dataset/no_person_in_water')

path, dirs, files = next(os.walk(videos_path))
file_count = len(files)

# define polygon points
points = np.array( [[[115, 288], [50,100], [0,100], [0,288]]], dtype=np.int32 )

images = []
masks = []

for video_number in range(file_count-1):
    video_path = videos_path + "/video_{}.mp4".format(video_number)
    segmentation_path = segmentations_path + "/segmentation_{}.mp4".format(video_number)
    
    video_cap = cv2.VideoCapture(video_path)
    segmentation_cap = cv2.VideoCapture(segmentation_path)
    
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_number%100 == 0:
        print("video number {}".format(video_number))
    
    frames = []
    for frame in range(0, frame_count, int(frame_count/2)):
        video_ret, video_frame = video_cap.read()
        segmentation_ret, segmentation_frame = segmentation_cap.read()
        
        video_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        segmentation_gray = cv2.cvtColor(segmentation_frame, cv2.COLOR_BGR2GRAY)

        video_poly = video_gray.copy()
        cv2.polylines(video_poly, [points], True, (0,0,255), 1)

        mask = np.zeros_like(video_gray)
        cv2.fillPoly(mask, [points], (1))
        
        image = video_gray * mask
        image = image[100:,:115]
        frames.append(image)
        
    segmentation_values = segmentation_gray[np.where(mask == 1)]
    
    person_in_water = sum(segmentation_values/255) > 15
    
    try:
        rgb = np.dstack((frames[0], frames[1], frames[2]))
        rgb = rgb.astype(np.uint8)

        if person_in_water:
            matplotlib.image.imsave('mini_project_dataset/person_in_water/image_{}.png'.format(video_number), rgb)
        else:
            matplotlib.image.imsave('mini_project_dataset/no_person_in_water/image_{}.png'.format(video_number), rgb)
    except:
        print("fuuuuuck")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




