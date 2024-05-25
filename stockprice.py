import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv(r'E:\迅雷下载\datasets-master\all_stocks_5yr.csv')

data['date'] = pd.to_datetime(data['date'])
companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NFLX', 'IBM', 'GE']
# date vs open
# date vs close
# plt.figure(figsize=(15, 8))
# for index, company in enumerate(companies, 1):
# 	plt.subplot(3, 3, index)
# 	c = data[data['Name'] == company]
# 	plt.plot(c['date'], c['close'], c="r", label="close", marker="+")
# 	plt.plot(c['date'], c['open'], c="g", label="open", marker="^")
# 	plt.title(company)
# 	plt.legend()
# 	plt.tight_layout()
# plt.show()

# plt.figure(figsize=(15, 8))
# for index, company in enumerate(companies, 1):
#     plt.subplot(3, 3, index)
#     c = data[data['Name'] == company]
#     plt.plot(c['date'], c['volume'], c='purple', marker='*')
#     plt.title(f"{company} Volume")
#     plt.tight_layout()
#     plt.show()


csco = data[data['Name'] == 'CSCO']
prediction_range = csco.loc[(csco['date'] > datetime(2013, 1, 1))
                             & (csco['date'] < datetime(2018, 1, 1))]
# plt.plot(csco['date'],csco['close'])
# plt.xlabel("Date")
# plt.ylabel("Close")
# plt.title("csco Stock Prices")
# plt.show()
close_data = csco.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
# prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary

model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(x_train,
                    y_train,
                    epochs=10)

# create test data
test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

train = csco[:training]
test = csco[training:]
test['Predictions'] = predictions

# Import necessary libraries
import matplotlib.pyplot as plt

# Plotting the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['close'], label='Train')
plt.plot(test['date'], test['close'], label='Test (Actual)')
plt.plot(test['date'], test['Predictions'], label='Test (Predicted)')
plt.title('CISCO SYSTEM Stock Close Price Prediction')
plt.xlabel('date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
