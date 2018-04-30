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
all_inputs = df[['sepal_length','sepal_width','petal_lenght','petal_width']].values
all_clases = df[['species']].values

tree.train
# Build a decision tree classifier to classify this dataset
clf = tree.DecisionTreeClassifier()

