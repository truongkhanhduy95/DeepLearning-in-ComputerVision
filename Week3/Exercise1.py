import numpy as np
from matplotlib import pyplot as plt
import cv2

def displayGrayscaleImage(image):
    plt.imshow(image,cmap='gray')
    plt.show()
    return image
    
imagePath = '/Users/duytruong/Desktop/Project/Personal Github/DeepLearning-in-ComputerVision/Week3/picture/dog.jpg'
img = cv2.imread(imagePath,0)
displayGrayscaleImage(img)

edges = cv2.Canny(img,100,200)
displayGrayscaleImage(edges)