import numpy as np
from matplotlib import pyplot as plt
import cv2
import urllib.request as req

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
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


# Download any color image from Internet and save it to your computer
imgurl ="https://www.codeproject.com/KB/GDI-plus/ImageProcessing2/img.jpg"
filePath = "picture/cute_dog.jpg"
# req.urlretrieve(imgurl, filePath)

# Convert the downloaded image to a grayscale image
img = cv2.imread(filePath,1)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(grayImg,cmap='gray')
plt.show()


# Apply watershed algorithm
ret,thresh = cv2.threshold(grayImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
fg = cv2.erode(thresh,None,iterations = 2)
bgt = cv2.dilate(thresh,None,iterations = 3)
ret,bg = cv2.threshold(bgt,1,128,1)
marker = cv2.add(fg,bg)
marker32 = np.int32(marker)
cv2.watershed(img,marker32)
m = cv2.convertScaleAbs(marker32)

plt.imshow(m)
plt.show()

ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img,mask = thresh)

plt.imshow(res)
plt.show()

