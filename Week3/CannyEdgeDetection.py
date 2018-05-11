# !pip install opencv-contrib-python

import numpy as np
from matplotlib import pyplot as plt
import cv2

def displayGrayscaleImage(image):
    plt.imshow(image,cmap='gray')
    plt.show()
    return image
    
def nothing(x):
  pass
    
imagePath = 'picture/dog.jpg'

img = cv2.imread(imagePath,0)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('t1', 'image', 0, 255, nothing)
cv2.createTrackbar('t2', 'image', 0, 255, nothing)


while(True):
    t1 = cv2.getTrackbarPos('t1', 'image')
    t2 = cv2.getTrackbarPos('t2', 'image')
    edges = cv2.Canny(img,t1,t2,L2gradient=False)
    cv2.imshow('image',edges)
    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
