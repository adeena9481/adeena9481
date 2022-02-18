# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 20:23:32 2022

@author: AD20094009
"""

import pandas as pd
import numpy as np
# import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
def initialize_weight_bias(dimension):
    w = np.full((dimension,1), 0.01)    # Create a matrix by the size of (dimension,1) and fill it with the values of 0.01
    b = 0.0
    return w,b


def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head



def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation:
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    
    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)     # loss function formula
    cost = (np.sum(loss)) / x_train.shape[1]                               # cost function formula
    
    # backward propagation:
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {'derivative_weight': derivative_weight, 'derivative_bias': derivative_bias}
    
    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, nu_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # Initialize for-back propagation for the number of iteration times. Then updating w and b values and writing the cost values to a list:  
    for i in range(nu_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
    
        # Update weight and bias values:
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        # Show every 20th value of cost:
        if i % 20 == 0:
            cost_list2.append(cost)
            index.append(i)
            print('Cost after iteration %i: %f' %(i,cost))
    
    parameters = {'weight': w, 'bias':b}
    
    # Visulization of cost values:
    plt.plot(index, cost_list2)
    plt.xlabel('Nu of Iteration')
    plt.ylabel('Cost Function Value')
    plt.show()
    
    return parameters, gradients, cost_list

def prediction(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
            
    return y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, nu_of_iteration):
    dimension = x_train.shape[0]
    w, b = initialize_weight_bias(dimension)    # Creating an initial weight matrix of (x_train data[0] x 1)
    
    # Updating our w and b by using update method. 
    # Update method contains our forward and backward propagation.
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, nu_of_iteration)
    
    # Lets use x_test for predicting y:
    y_test_predictions = prediction(parameters['weight'], parameters['bias'], x_test)
    
    with open("weight.pkl","wb") as f:
        pickle.dump(parameters['weight'], f)
    with open("bias.pkl","wb") as f2:
        pickle.dump(parameters['bias'], f2)
    # Investigate the accuracy:
    print('Test accuracy: {}%'.format(100 - np.mean(np.abs(y_test_predictions - y_test))*100))
# read the cleaned data
data = pd.read_csv("weatherAUS.csv")
data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'    ], axis=1, inplace=True)
# data.head(5)
data.fillna(data.mean(), inplace=True)
# the features or the 'x' values of the data
# these columns are used to train the model
# the last column, i.e, precipitation column
# will serve as the label
data.RainToday = [1 if each == 'Yes' else 0 for each in data.RainToday]
data.RainTomorrow = [1 if each == 'Yes' else 0 for each in data.RainTomorrow]
# data.sample(3)
# the output or the label.
# Y = data['PrecipitationSumInches']
# # reshaping it into a 2-D vector
# Y = Y.values.reshape(-1, 1)
y = data.RainTomorrow.values
x_data = data.drop('RainTomorrow', axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=75)
# # consider a random day in the dataset
# we shall plot a graph and observe this
# day
# day_index = 798
# days = [i for i in range(Y.size)]

# initialize a linear regression classifier
# clf = LinearRegression()
# x_train=X.iloc[:int(len(X)*0.7),:]
# x_test=X.iloc[int(len(X)*0.7):,:]
# y_train=Y[:int(len(X)*0.7)]
# y_test=Y[int(len(X)*0.7):]
logistic_regression(x_train.T, y_train.T, x_test.T, y_test.T, learning_rate=1, nu_of_iteration=400)

with open("weight.pkl","rb") as f:
    zz=pickle.load(f)
with open("bias.pkl","rb") as f:
    zz2=pickle.load(f)
#########Pridiction code######################

input2={'MinTemp':[0.36320754716981135],
 'MaxTemp':  [0.4839319470699433],
 'Rainfall': [0.0],
 'Evaporation':[0.027586206896551724],
 'Sunshine':[0.5249087945283548],
 'WindGustSpeed':[0.20930232558139536],
 'WindSpeed9am':[0.11538461538461539],
 'WindSpeed3pm':[0.21839080459770116],
 'Humidity9am':[70.39],
 'Humidity3pm': [0.22],
 'Pressure9am':[0.7933884297520661],
 'Pressure3pm':[0.7696000000000017],
 'Cloud9am': [0.0],
 'Cloud3pm': [0.0],
 'Temp9am': [0.45569620253164556],
 'Temp3pm': [0.4894433781190019],
 'RainToday': [0.0]}

test=pd.DataFrame(input2)
y_test_predictions = prediction(zz, zz2, test.T)

print(y_test_predictions.T)

