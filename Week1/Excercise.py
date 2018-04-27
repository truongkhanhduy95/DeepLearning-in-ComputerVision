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

def display_image(img, n_channel=3):
    if n_channel == 4:
        img_bgr = img[:, :, :3]
        img_alpha = img[:, :, 3]
        img_white = np.ones_like(img_bgr, dtype=np.uint8) * 255 # An image with all white
        alpha_factor = img_alpha[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

        # Display the color image
        img_base = img_bgr.astype(np.float32) * alpha_factor
        img_white = img_white.astype(np.float32) * (1 - alpha_factor)
        img = (img_base + img_white).astype(np.uint8)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

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
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.show()
img_new = display_image(img, 4)

#Display image
# cv2.imshow('image',bin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
