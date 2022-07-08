# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:02:50 2021

@author: correa
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import math

#Cropping into Imagettes
def crop_Img_BySide(img = None ,  _width = 256 ,_height = 256):
    
    # img : IMAGE TO BE CUT
    # _height : desired height
    # _width : desired width
    
    h , w , c = img.shape
    
    limitHeight = int(h/_height)
    limitWidth = int(w/_width)
    

    coordHeight = [x *_height  for x in list(range(0,limitHeight+1,1))]
    
    coordWidth = [x * _width  for x in list(range(0,limitWidth+1,1))]
   
    imgC=img.copy()

    imagettes = []

    for i in range(len(coordWidth)-1):
           
        for j in range(len(coordHeight)-1):

            imagettes.append(imgC[coordHeight[j]:coordHeight[j+1],coordWidth[i]:coordWidth[i+1]])

    if (w/_width).is_integer():  
        pass  
    else:

        for j in range(len(coordHeight)-1):     
            #print("debug 1")
            imagettes.append(imgC[ coordHeight[j]:coordHeight[j+1]  , w - _width : w])
               
    if (h/_height).is_integer():
        pass  
    else: 

        for i in range(len(coordWidth)-1):
            #print("debug 2")
            imagettes.append(imgC[h - _height : h , coordWidth[i]:coordWidth[i+1] ])

    return imagettes

#Building predicted image
def img_Reconstruction(_imagettes, img):
    
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #Original size
    h , w    = img.shape
    hi , wi = _imagettes[0].shape
    
    numpy = np.zeros((h,w),dtype='uint8')
    
    reportH = int(h/hi)
    reportW = int(w/wi)
    
    
    coordHeight = [x *hi  for x in list(range(0,reportH+1,1))]
    
    coordWidth = [x * wi  for x in list(range(0,reportW+1,1))]
    
    k=0
    
    for i in range(len(coordWidth)-1):
       for j in range(len(coordHeight)-1):

           numpy[coordHeight[j]:coordHeight[j+1],coordWidth[i]:coordWidth[i+1]]= _imagettes[k]
           k=k+1
           
           
    if (w/wi).is_integer():  
        pass  
    else:

        for j in range(len(coordHeight)-1):     
            #print("debug 1")
            numpy[coordHeight[j]:coordHeight[j+1],w - wi : w]= _imagettes[k]
            k=k+1 
           
    if (h/hi).is_integer():
        pass  
    else: 

        for i in range(len(coordWidth)-1):
            #print("debug 2")
            numpy[ h-hi : h , coordWidth[i]:coordWidth[i+1]]= _imagettes[k]
            k=k+1  
    

    return numpy


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y



def scaleExtractor(img):
    
    
    h,w,c = img.shape
    #We take just the label
    cropped_image = img[h-128:h, w-256:w]
    
    img = cropped_image.copy()
    
    gray = cv.cvtColor(cropped_image,cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 254, 255, cv.THRESH_BINARY)
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # filter noisy detection
    contours = [c for c in contours if cv.contourArea(c) > 100]
    
    # sort from by (y, x)
    contours.sort(key=lambda c: (cv.boundingRect(c)[1], cv.boundingRect(c)[0]))
    # work on the segment
    
    cv.rectangle(cropped_image, cv.boundingRect(contours[-1]), (0,255,0), 2)

    x,y,w,h = cv.boundingRect(contours[-1])
    
    
    #print(x,y,w,h) # x,y: (39 152) w,h: [304 21]
    
    #Note: Remember that the Y coordinate in OpenCV, starts from the bottom of the image to the top of the image.
    
    imgCrop2 = img[y:y+h, x : x+w]
        
    #cv.imshow("Detection",img)
    
    #cv.imshow("Final CROP",imgCrop2)
    
    gray = cv.cvtColor(imgCrop2,cv.COLOR_BGR2GRAY)
    
    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    low_threshold = 50
    high_threshold = 150
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi /2  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    
    line_image = imgCrop2.copy()  # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    #points = []
    dist = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            #points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            dist.append(abs(x2-x1))
      
    #print(f"Scale ----> {max(dist)} pixels ==")
    
    cv.imwrite("edges.png",edges) 
    cv.imwrite("lines.png",line_image)
  

    cv.waitKey(0)
    cv.destroyAllWindows() 
    

    return max(dist)