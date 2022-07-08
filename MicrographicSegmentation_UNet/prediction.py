# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:50:16 2021

@author: correa
"""

from keras.models import load_model
from outils import *


def makePredictions(imagettes,img):
    
    model = load_model('model_test3_resnet50_UNet_crossEntropy_epoch100_train160_val40.h5', compile=False)
    
    predictions = []
    
    for i in range(len(imagettes)):
        #cv.imshow('imagette',imagettes[0])
    
        testImg = np.expand_dims(imagettes[i], axis=0)
        
        prediction = model.predict(testImg)
        
        predicted_img = np.argmax(prediction, axis=3)[0,:,:]
        
        #convertion du format pour affichage
        predicted_img=np.array(predicted_img, dtype = np.uint8 )
        
        predictions.append(predicted_img)
    
    
    
    
    imgSegmentated = img_Reconstruction(predictions, img)
    
    print("type of segmentated image: ", imgSegmentated.dtype)
    
    print("Shape of segmentated image:",imgSegmentated.shape)
    
    print("unique seg",np.unique(imgSegmentated))
          
    #imgGray = cv.cvtColor(imgSegmentated, cv.COLOR_BGR2GRAY)  
    
    
    #mask =np.zeros (img.shape,img.dtype)
    #print(np.unique(imgGray))
    
    img  = imadjust(imgSegmentated, 0,3 , 0 ,255)
    
    img  = img.astype('uint8')
    
    print(np.unique(img))
    
    return img