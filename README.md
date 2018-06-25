# Deep Learning in Computer Vision

## Week 0: Linear Algebra and Calculus

Exercise: 
Given that x is a vector in R3 and the tail of x is at the origin. Where is the locus of the tip of vector x such that the norm L1 of x is equal to 4. What is the focus if x is a vector in R4 ?

## Week 1: Image Processing in Computer Vision

Exercise: 
1. Download any color image file with PNG format from Internet (for those who have no idea about PNG file, please see the link: https://en.wikipedia.org/wiki/Portable_Network_Graphics )

2. Write program to do the following works:
	* Load the color image file downloaded in Step 1
	* Display the color image
	* Convert the color image to a gray image, save to a file
	* Reload the file with gray image and display
	* Make sure your saved files can be opened and displayed by other programs, e.g., ImageViewer, Photoshop etc

3. Take a break

4. Write the program to do the following works:
	* Reload the color image file downloaded in Step 1
	* Resize the image to the size of 256 (pixels) x 256 (pixels)
	* Display the image
	* Save to a file
	* Reload the gray image file converted in Step 2
	* Resize the gray image to the  size of 256 (pixels) x 256 (pixels)
	* Display the image
	* Save to a file

5. Write the program to do the following works:
	* Apply Gaussian filter with different kernel sizes and sigma
	* Explain the differences

6. Use the perspective projection equations to explain why, in a picture of a face taken frontally and from a very small distance, the nose appears much larger than the rest of the face. Can this effect be reduced by acting on the focal length? 

## Week 2: Traditional Machine Learning

Exercise: 
1. Download the Iris flower data set (https://en.wikipedia.org/wiki/Iris_flower_data_set)

2. Write program to complete the following works:
	* Visualize the dataset
	* Build a decision tree classifier to classify this dataset
	* Modify parameters/hyper-parameters to get the best result

3. Take a break

4. Write the program to do the following works:
	* Build a SVM classifier to classify this dataset
	* Modify parameters/hyper-parameters to get the best result

5. What’s the trade-off between bias and variance?
6. What is the difference between supervised and unsupervised machine learning?
7. How is KNN different from k-means clustering?

## Week 3: Edge Detection, Image Tracking

Exercise: 
1. Download any color image from Internet and save it to your computer

2. Write program to complete the following works:
	* Convert the downloaded image from 1. to a grayscale image
	* Apply Canny Edge detector to the grayscale image with fixed threshold as your choice 		(https://en.wikipedia.org/wiki/Canny_edge_detector)
	* Write a small application to find the Canny edge detection whose threshold values can be varied using two trackbars

3. Take a break

4. Review the watershed algorithm (https://en.wikipedia.org/wiki/Watershed_(image_processing))

5. Write program to complete the following works:
	* Download any color image from Internet and save it to your computer
	* Convert the downloaded image to a grayscale image
	* Apply watershed algorithm to the above image and observe outputs with different parameters
	
You can refer to the following to understand more about this algorithm: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html.

## Week 4: Tensorflow/Keras

Exercise: 
1. Install Tensorflow. Instruction: https://www.tensorflow.org/install/	
2. Perform image recognition using pre-trained models. Instruction: https://www.tensorflow.org/tutorials/image_recognition
Try different networks:
	* AlexNet
	* VGG
	* Inception
	* ResNet

## Week 5: Tensorflow exercise

Exercise: 
Using Google Colab, open the Jupiter notebook file, try to modify each layer's hyperparameters,	dropout connection between layers to get better model.


## Week 6: Object detection / Sematic segmentation

## Week 7: Transfer learning

## Week 8: Applications

## Week 9: Transfer learning tutorial

Lecture URL: https://www.youtube.com/watch?v=2XKh4BtzUMo
