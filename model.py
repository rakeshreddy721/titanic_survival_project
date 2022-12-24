# -*- coding: utf-8 -*-

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# importing the data file into a dataframe
data = pd.read_csv('titanic_dataset.csv')

# dropping the non-essential columns from the dataframe
data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# label encoding the only categorical variable 'Sex'
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
# from above encoding, male encoded to 1, female encoded to 0

# data.info() shows the column 'Age' has the null values
# However, below plot shows there is not much correlation between Age and Survival
sns.boxplot(x='Survived', y='Age', data=data, palette='plasma')
print('The correlation between Age and Survived features is: ' + str(data['Age'].corr(data['Survived'])))

# Hence, imputing the missing values with 150
data['Age'].fillna(150, inplace=True)

# verifying the correlation again after imputing the missing values with 150.
# It didn't change much. So we are good.
print('The correlation between Age and Survived features is: ' + str(data['Age'].corr(data['Survived'])))

# Age has many unique and conitnious values. We can bin these values
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 200]
bin_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-200']
data['Age_bin'] = pd.cut(data['Age'], bins=bins, labels=bin_labels)

# Dropping Age feature
data.drop('Age', axis=1, inplace=True)

# since new feature 'Age_bin' is ordinal category feature we perform one-hot encoding using pd.get_dummies
data = pd.get_dummies(data, columns=['Age_bin'], drop_first=True)

# # Fare column is having high values different from the dataset. Lets normalize the column
# scaler = StandardScaler()
# data['Fare'] = scaler.fit_transform(np.reshape(np.array(data['Fare']),(-1,1)))

# # Standardizing the Pclass feature
# scaler2 = StandardScaler()
# data['Pclass'] = scaler2.fit_transform(np.reshape(np.array(data['Pclass']),(-1,1)))

# Separating the dataset into Independent and dependent dataframes
X = data.drop('Survived', axis=1)
y = data['Survived']

# # --------------------
# # split the data into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # model the train data
# model = tree.DecisionTreeClassifier()
# # model = LogisticRegression(max_iter=1000)
# # model = SVC()
# # model = GaussianNB()
# model.fit(X_train, y_train)
#
# # predict the outputs for test inputs
# y_hat = model.predict(X_test)
#
# # check the accuracy
# accuracy = accuracy_score(y_test, y_hat)
#
# print(accuracy)
# # --------------------

# create instance of DecisionTreeClassifier model
model_dec_tree = tree.DecisionTreeClassifier()

# fit model with data
model_dec_tree.fit(X,y)

# Save model to disk
pickle.dump(model_dec_tree,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,1,20,0,0,0,0,1,0,0,0,0]]))
