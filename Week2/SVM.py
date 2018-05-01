# Import packages
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

# Build a decision tree classifier to classify this dataset
