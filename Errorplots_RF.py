# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:07:43 2018

@author: Rajesh
"""

# Import the Libararies.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# Import the Datasets.
dataset = pd.read_csv('IOC.csv')
date = pd.DataFrame(dataset.iloc[:, 2].values)
date_test = pd.DataFrame(date.iloc[42:64, ].values)
X = pd.DataFrame(dataset.iloc[:, 2:8].values)
y = dataset.iloc[:, 8].values
mse = []
r2 = []
mae = []
l1 = []
trees = np.array([50, 100, 150, 200])

# Label the date.
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])

# Split the training set and test set.
X_train = pd.DataFrame(X.iloc[0:42, ].values)
y_train = y[0:42, ]
X_test = pd.DataFrame(X.iloc[42:64, ].values)
y_test = y[42:64, ]

# Scale the features.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

for i in range(4):
    
    # Train the Randomforest model.
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = trees[i], random_state = 0)
    regressor.fit(X_train, y_train)
    
    # predict the output.
    Randomforest_y_pred = regressor.predict(X_test)
    
    # Calculate Mean Squared Error.
    from sklearn.metrics import mean_squared_error
    RF_error = mean_squared_error(y_test, Randomforest_y_pred)
    mse.append(RF_error)
    
    # R^2 error
    from sklearn.metrics import r2_score
    RF_r2 = r2_score(y_test, Randomforest_y_pred)
    r2.append(RF_r2)
    
    # Median Absolute Error
    from sklearn.metrics import median_absolute_error
    RF_mae = median_absolute_error(y_test, Randomforest_y_pred)
    mae.append(RF_mae)
    
    # L1 norm 
    RF_norm = np.linalg.norm((y_test - Randomforest_y_pred), ord = 2)
    l1.append(RF_norm)
  
# Plot of Mean Squared error.    
plt.plot(trees, mse)
plt.xlabel('Number of trees')
plt.ylabel('MSE')
plt.title('Plot between number of trees vs MSE')
plt.show()

# Plot of Median Absolute error.
plt.plot(trees, mae)
plt.xlabel('Number of trees')
plt.ylabel('MAE')
plt.title('Plot between number of trees vs MAE')
plt.show()

# Plot of R-Squared error.
plt.plot(trees, r2)
plt.xlabel('Number of trees')
plt.ylabel('R-Squared')
plt.title('Plot between number of trees vs R-Squared')
plt.show()

# Plot of L1-Norm
plt.plot(trees, l1)
plt.xlabel('Number of trees')
plt.ylabel('L1-Norm')
plt.title('Plot between number of trees vs L1-Norm')
plt.show()
   