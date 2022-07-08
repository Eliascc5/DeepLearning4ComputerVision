# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:37:56 2021

@author: correa
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from outils import *
import scipy.stats as st
import seaborn as sns


SCALE = 156

path = 'Acier_P265GH_x200_segmentated.png'
img = cv.imread(path)
h,w,c=img.shape

areaImg_pixels= h*w

areaImg_um = (h/SCALE)*50 *(w/SCALE)*50
print("area in um:",areaImg_um)

imgColor=img.copy()

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

classes = np.unique(img)
print("shape of image",img.shape)
print("valeus of pixels",classes)

maskClasses = []

for i in range(len(classes)):
    
    mask = np.zeros(imgGray.shape,imgGray.dtype)
    classIndex = np.where(np.equal(imgGray, classes[i]))
    mask[classIndex] = np.array([255])
    maskClasses.append(mask)

cv.imshow("mask with black",maskClasses[2])

#cv.imwrite("mask1.png",maskClasses[0])
#cv.imwrite("mask2.png",maskClasses[1])
#cv.imwrite("mask3.png",maskClasses[2])
#cv.imwrite("mask4.png",maskClasses[3])



contours, hierarchy = cv.findContours(maskClasses[2],cv.RETR_TREE ,cv.CHAIN_APPROX_NONE) 


print("Contours", len(contours))

countValid = []

areaGrainBlack = []



for i , count in enumerate(contours):
    area = cv.contourArea(count)
    if area > 10 :
      countValid.append(count)
      
      areaGrainBlack.append(area)
      
      
for i in range(len(areaGrainBlack)):     
    areaGrainBlack[i] = areaGrainBlack[i]*(areaImg_um/areaImg_pixels)

print("counts valid", len(countValid))
print("max", max(areaGrainBlack))

cv.fillPoly(imgColor, pts = countValid, color=(0,255,0))

cv.imwrite("imgColor_mask4.png",imgColor)

cv.imshow('result',imgColor)





cv.waitKey(0)
cv.destroyAllWindows()

sns.displot(areaGrainBlack, bins=10,kde=False)

plt.title("Distribution de taille des grains")
plt.xlabel("Surface en µm carrés")
plt.ylabel("Nombre de grains")
plt.show()