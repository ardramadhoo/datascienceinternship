# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 08:17:29 2022

@author: Ardra
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rent = pd.read_csv('rent.csv')

data = rent.copy()

#Exploratory data analysis:

print('Dataset')
data.head()

print('Column descriptions')
data.info()

#**** Check for missing values    
print("Columnwise missing values")
print(data.isnull().sum())
""" There are no missing values """

print("Summary of numerical variables")
data.describe()

print("Summary of categorical variables")
data.describe(include='O')

data2 = pd.read_csv('rent.csv',na_values=['?'])
#will give the same dataframe as no missing values

# Dropping unwanted columns

col=['Posted On']
data2 = data2.drop(columns=col, axis=1)

# Realtionship between independent variables
correlation = data2.corr()

fig,ax = plt.subplots(3,3,figsize=(18,15))
fig.suptitle('DATA VISUALIZATIONS')

sns.distplot(data2.Rent,ax=ax[0,0])
ax[0,0].set_title('Plot of Rent')

sns.distplot(data2.Size,ax=ax[0,1])
ax[0,1].set_title('Plot of Size')

sns.regplot(ax=ax[0,2],x='Size',y='Rent',data=data2,color='red',fit_reg=False)
ax[0,2].set_title('Rent vs Size')
"""Houses of higher size have higher price"""

sns.countplot(ax=ax[1,0],x='Furnishing Status',data=data2)
ax[1,0].set_title('Furnishing Status')

sns.countplot(ax=ax[1,1],x='Area Type',data=data2)
ax[1,1].set_title('Area Type')
"""Built Area Type - insignificant"""

sns.countplot(ax=ax[1,2],x='City',data=data2)
ax[1,2].set_title('City')

sns.countplot(ax=ax[2,0],x='Tenant Preferred',data=data2)
ax[2,0].set_title('Tenant Preferred')

sns.countplot(ax=ax[2,1],x='Bathroom',data=data2)
ax[2,1].set_title('Rent vs Bathrooms')

sns.countplot(ax=ax[2,2],x='Point of Contact',data=data2)
ax[2,2].set_title('Point of Contact')
"""Contact Builder insignificant"""
plt.show()

sns.pairplot(data2,hue='Furnishing Status')
plt.show()

# Removing insignificant variables

col=['Point of Contact','Area Locality',
     'City','Furnishing Status','Tenant Preferred','Floor']
data3 = data2.drop(columns=col, axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data3 = data3.dropna(axis=0)
# Converting categorical variables to dummy variables
data3 =pd.get_dummies(data3,drop_first=True) 

# Separating input and output features
x1 = data3.drop(['Rent'], axis='columns', inplace=False)
y1 = data3['Rent']

# Plotting the variable rent
r = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
r.hist()
plt.show()

# Transforming rent as a logarithmic value
y1 = np.log(y1)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lgr=LinearRegression(fit_intercept=True)

model_lin1=lgr.fit(X_train,y_train)

rent_predictions_lin1 = lgr.predict(X_test)

lin_mse1 = mean_squared_error(y_test, rent_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)

residuals1=y_test-rent_predictions_lin1
plt.title('Residuals plot')
sns.regplot(x=rent_predictions_lin1, y=residuals1, scatter=True, 
            fit_reg=False,color='green')
plt.show()
print(residuals1.describe())

print("R squared value for train from Linear Regression=  %s"% r2_lin_train1)
print("R squared value for test from Linear Regression=  %s"% r2_lin_test1)
print("RMSE value for test from Linear Regression=  %s"% lin_rmse1)