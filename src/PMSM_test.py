import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12,5)
plt.style.use('fivethirtyeight')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset into a pandas DataFrame
df = pd.read_csv('input/measures_v2.csv')

# View the first few rows of the dataset
df.head(5).style.set_properties(**{'background-color':'lightgreen','color':'black','border-color':'#8b8c8c'})

# Check the shape of the dataset
print(df.shape)

# Check the data types of the columns
print(df.dtypes)

# Generate descriptive statistics of the dataset
print(df.describe())

# Plot a histogram of the "motor_speed" column
sns.histplot(df['motor_speed'], kde=False)
plt.show()

# Plot a scatterplot of the "motor_speed" vs "torque" columns
sns.scatterplot(x='motor_speed', y='torque', data=df)
plt.show()

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

