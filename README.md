this readme is writen by chatgpt 4.0, idk how to write a useful readme :(
# Cisco Stock Price Prediction

This project uses an LSTM (Long Short-Term Memory) neural network to predict Cisco stock prices based on historical data. The model is implemented using TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Predicting stock prices is a complex task due to the volatile nature of financial markets. This project leverages the power of LSTM networks to make predictions based on historical stock price data. The model is trained to predict the closing prices of Cisco Systems, Inc. (CSCO) stock.

## Dataset
The dataset used in this project consists of historical stock prices for various companies over a 5-year period. The data includes open, high, low, close, volume, and name of the stock.

- **Source:** [Kaggle - All Stocks 5 Year Data](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)

## Installation
To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
