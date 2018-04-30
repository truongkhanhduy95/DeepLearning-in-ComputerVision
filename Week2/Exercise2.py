# Import packages
from sklearn import tree
from sklearn import cross_validation
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

# Visualize the dataset
df = pd.read_csv('dataset/iris.csv')
print(df)

# Explore  data
df.isnull().any()
df.dtypes
df.describe()
df['petal_width'].plot.hist()
plt.show()
sns.pairplot(df, hue='species')

# Split dataset into training data and testing data 
all_inputs = df[['sepal_length','sepal_width','petal_length','petal_width']].values
all_classes = df[['species']].values

(train_inputs, test_inputs, train_classes, test_classes) = cross_validation.train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

# Build a decision tree classifier to classify this dataset
dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
dtc.score(test_inputs, test_classes)
