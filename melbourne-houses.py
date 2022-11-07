# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:29:29 2022

@author: shrey
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

""" This dataset contains features of houses in Melbourne and their prices.
    I am going to be using the scikit-learn package to predict how much a house
    in Melbourne costs, given the features."""

# Reading in the Melbourne houses data set

mel_data = pd.read_csv(r'C:\Users\shrey\OneDrive\Desktop\coding trials\melb_data.csv')
pd.set_option('max_rows', 6)
pd.set_option('max_columns', 5)

list_of_col = list(mel_data.columns)

print(list_of_col)

# For this current model, I'm going to be only using numerical features.
# Choosing columns that only have numerical values (calling it X)
# and dropping rows with missing values.

num_feat = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
mel_data = mel_data.dropna(axis=0)
X = mel_data[num_feat]

# Our target variable is price. This is what I'm trying to predict.

y = mel_data.Price

# Splitting our data into training and validation data

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0, train_size = 0.5)

# To deal with the missing values in cells, SimpleImputer() is used.

imp = SimpleImputer(strategy = 'mean')

imp_train_X = pd.DataFrame(imp.fit_transform(train_X))
imp_test_X = pd.DataFrame(imp.transform(test_X))

# Imputation removes the columns so they're put back in

imp_train_X.columns = train_X.columns
imp_test_X.columns = test_X.columns

# Introducing the type of model 

mel_model = DecisionTreeRegressor(max_leaf_nodes = 100)

# Fitting the data to the model

mel_model.fit(train_X, train_y)
y_pred = mel_model.predict(test_X)

# Printing a house's predicted value and its real value.

print('The 135th house\'s real value is {real}. Its predicted value is {pre}'.format(pre = y_pred[135], real = list(test_y)[135]))

""" This is a relativity good prediction but this can not be generalised for the all the houses """
# Finding the average of all the errors between the houses' real and predicted prices.

mae = mean_absolute_error(y_pred, test_y)

print(mae)

""" This is quite a high number so this model is not great at estimating house prices"""

# Can introduce random forests and see if this will result in more accurate predictions.

from sklearn.ensemble import RandomForestRegressor

# Introducing the type of model and fitting the data

forest_mel_model = RandomForestRegressor(random_state = 0)
forest_mel_model.fit(train_X, train_y)
y_pred_forest = forest_mel_model.predict(test_X)

mae_forest = mean_absolute_error(y_pred_forest, test_y)

print(mae_forest)

""" This seems to be a better prediction but still not a huge improvement."""






















