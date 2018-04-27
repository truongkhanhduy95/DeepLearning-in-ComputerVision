# Install packages
! pip install --upgrade pip
! pip install pandas
! pip install --upgrade numpy
! pip install opencv-python
! pip install --upgrade opencv-python

# Import packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

def invertAlpha(img):
    base_img = img[:, :, :3]
    alpha_img = img[:, :, 3]
    white_img = np.ones_like(base_img, dtype=np.uint8) * 255 
    alpha_factor = alpha_img[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), 2)

    img_base = base_img.astype(np.float32) * alpha_factor
    white_img = white_img.astype(np.float32) * (1 - alpha_factor)
    img = (img_base + white_img).astype(np.uint8)
    return img

def display(img):
    plt.imshow(cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB))
    plt.show()
    
#Load image from folder
img = cv2.imread('/Users/truongkhanhduy/Desktop/Project/Personal Github/DeepLearning-in-ComputerVision/Week1/images/image1.png',-1)

# alpha = img[:,:,3] #extract alpha channel
# print(alpha)
# bin = ~alpha #invert b/w
# print(bin)

# new = cv2.merge((img[:,:,0],img[:,:,1],img[:,:,2],bin))
# new[:,:,3] = bin
# print(new)
# bin[:,:,0] = img[:,:,0]
#RGB_img = 

#Display image
display(invertAlpha(img))

#Convert the color image to a gray image, save to a file
gray_img = cv2.cvtColor(invertAlpha(img), cv2.COLOR_BGR2GRAY)
path = "/Users/truongkhanhduy/Desktop/Project/Personal Github/DeepLearning-in-ComputerVision/Week1/images/gray_image.png"
cv2.imwrite(path, gray_img)
print("Write Gray-scale Image to file: ", path)

#Display gray image
img_gray_display = cv2.imread(path, cv2.IMREAD_UNCHANGED)
display(img_gray_display)
