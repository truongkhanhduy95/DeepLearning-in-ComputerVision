# Import packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from sklearn import svm,cross_validation

# Visualize the dataset
df = pd.read_csv('dataset/iris.csv')
print(df)

# Split dataset into training data and testing data 
all_inputs = df[['sepal_length','sepal_width','petal_length','petal_width']].values
all_classes = df[['species']].values

(train_inputs, test_inputs, train_classes, test_classes) = cross_validation.train_test_split(all_inputs, all_classes, train_size=0.7)

# Build a SVM classifier to classify this dataset
model = svm.SVC()
model.fit(train_inputs, train_classes)

y_pred = model.predict(test_inputs)
print(model.score(test_inputs,test_classes.ravel())) #0.9555555555555556 Accuracy

# Modify parameters/hyper-parameters to get the best result
modify_model = svm.SVC(kernel='linear', degree=3)
modify_model.fit(train_inputs, train_classes)

y_pred = modify_model.predict(test_inputs)
print(modify_model.score(test_inputs,test_classes.ravel())) #0.9777777777777777 Accuracy

