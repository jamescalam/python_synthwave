# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:16:07 2020

@author: James
"""

import cv2
import os
import imageio

VIDEO = False
GIF = True

image_folder = '../visuals/0513'
video_name = '../visuals/0513_training_6fps.mp4'

images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg")
              and 'loss' not in img]

images2 = sorted([('epoch_'+'0'*(21-len(x))+x.replace('epoch_', ''), x) for x in images])

if VIDEO:
    frame = cv2.imread(os.path.join(image_folder, images2[0][1]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, 6, (width, height))
    
    for image in images2:
        video.write(cv2.imread(os.path.join(image_folder, image[1])))
    
    cv2.destroyAllWindows()
    video.release()
    
if GIF:
    gif = []
    for i, image in enumerate(images2):
        gif.append(imageio.imread(os.path.join(image_folder, image[1])))
        if i > 182:
            break
    imageio.mimsave('../visuals/0513_training.gif', gif)