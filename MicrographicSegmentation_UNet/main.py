# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:21:24 2021

@author: correa
"""



from outils import *
from prediction import makePredictions


path = 'Acier_P265GH_x200.jpg'
img = cv.imread(path)

imgColor = img.copy()
print("Shape of image:",img.shape)


#scale = scaleExtractor(img)
#print("Echelle:",scale )

imagettes = crop_Img_BySide(img,256,256)

imgSegmentated = makePredictions(imagettes,imgColor)


cv.imshow("after transformation", imgSegmentated)
cv.imwrite(f'{path}_segmentated.png', imgSegmentated)


cv.waitKey(0)
cv.destroyAllWindows()
