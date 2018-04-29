# Import packages
from urllib.request import urlretrieve
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2

iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)

df = pd.read_csv(iris, sep=',')

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df.columns = attributes

file_name = ''
df.to_csv(file_name, encoding='utf-8', index=False)

print(df)


