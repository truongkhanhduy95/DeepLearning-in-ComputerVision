# Install packages
! pip install --upgrade pip
! pip install pandas
! pip install numpy
! pip install opencv-python
! pip install --upgrade opencv-python

# Import packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

#Load image from folder
img = cv2.imread('/Users/duytruong/Desktop/Project/Personal Github/DeepLearning-in-ComputerVision/Week1/images/image1.png',-1)

# alpha = img[:,:,3] #extract alpha channel
# print(alpha)
# bin = ~alpha #invert b/w
# print(bin)

# new = cv2.merge((img[:,:,0],img[:,:,1],img[:,:,2],bin))
# new[:,:,3] = bin
print(new)
# bin[:,:,0] = img[:,:,0]
#RGB_img = 
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


#Display image
# cv2.imshow('image',bin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
