# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 20:15:03 2018

@author: Rajesh
"""
# Import the Libararies.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Import the Datasets.
dataset = pd.read_csv('IOC.csv')
date = pd.DataFrame(dataset.iloc[:, 2].values)
date_test = pd.DataFrame(date.iloc[42:64, ].values)
X = pd.DataFrame(dataset.iloc[:, 2:8].values)
y = dataset.iloc[:, 8].values

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

# Train the svm model
from sklearn.svm import SVR
SVM_regressor = SVR(kernel = 'rbf', gamma = 0.1)
%time SVM_regressor.fit(X_train, y_train)

# predict the output
SVM_y_pred = SVM_regressor.predict(X_test)

# Train the Randomforest model.
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
%time RF_regressor.fit(X_train, y_train)

# predict the output.
Randomforest_y_pred = RF_regressor.predict(X_test)

# Calculate Mean Squared Error.
from sklearn.metrics import mean_squared_error
RF_error = mean_squared_error(y_test, Randomforest_y_pred)
SVM_error = mean_squared_error(y_test, SVM_y_pred)

# R^2 error
from sklearn.metrics import r2_score
SVM_r2 = r2_score(y_test, SVM_y_pred) 
RF_r2 = r2_score(y_test, Randomforest_y_pred)

# Median Absolute Error
from sklearn.metrics import median_absolute_error
SVM_mae = median_absolute_error(y_test, SVM_y_pred)
RF_mae = median_absolute_error(y_test, Randomforest_y_pred)

# L1 norm 
SVM_norm = np.linalg.norm((y_test - SVM_y_pred), ord = 2)
RF_norm = np.linalg.norm((y_test - Randomforest_y_pred), ord = 2)

SVM_y_pred = pd.DataFrame(SVM_y_pred)
Randomforest_y_pred = pd.DataFrame(Randomforest_y_pred)
# Add Date, Actual Values, SVM predicted output, Random Forest output to output.
output = pd.DataFrame()
output['Date'] = date_test[0]
output['Actual'] = y_test
output['SVM_Prediction'] = SVM_y_pred[0]
output['Rf_prediction'] = Randomforest_y_pred[0]

# Plot between Actual Prices vs SVM predicted prices.
fig, ax = plt.subplots()
ax.plot(output['Date'], output['Actual'], color = 'blue', label = 'Actual Price')
ax.plot(output['Date'], output['SVM_Prediction'], color = 'red', label = 'SVM Predicted Price')
plt.legend(loc='lower right')
fig.autofmt_xdate(rotation = 90)
ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.set_title('Forecasting Stock Prices')
plt.xlabel('Date')
plt.ylabel('prices')

# Plot between Actual Prices vs Randomforest predicted prices.
fig, ax = plt.subplots()
ax.plot(output['Date'], output['Actual'], color = 'blue', label = 'Actual Price')
ax.plot(output['Date'], output['Rf_prediction'], color = 'green', label = 'RandomForest Predicted Price')
plt.legend(loc='lower right')
fig.autofmt_xdate(rotation = 90)
ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.set_title('Forecasting Stock Prices')
plt.xlabel('Date')
plt.ylabel('prices')

# Plot between Actual Prices vs SVM predicted prices vs Randomforest predicted prices.
fig, ax = plt.subplots()
ax.plot(output['Date'], output['Actual'], color = 'blue', label = 'Actual Price')
ax.plot(output['Date'], output['SVM_Prediction'], color = 'red', label = 'SVM Predicted Price')
ax.plot(output['Date'], output['Rf_prediction'], color = 'green', label = 'RandomForest Predicted Price')
plt.legend(loc='lower right')
fig.autofmt_xdate(rotation = 90)
ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.set_title('Forecasting Stock Prices')
plt.xlabel('Date')
plt.ylabel('prices')



