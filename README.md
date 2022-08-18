# timeseries-
A Simple Project about Sales Forecast of a Pesticide company
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the data
df=pd.read_excel(r"C:\Users\aalme\OneDrive\Desktop\PROJECT\ABC_manufacturing_processed_final.xlsx",header=0)


df.head()

df.dtypes

df["STATE"].value_counts()

df["COMPANY"].value_counts()

#Doing Statewise prediction
company_state =  df[df['STATE']=='Uttar Pradesh']

company_state.shape

company_state.head()

company_state =company_state[['DATE_1','VALUE_x']]

company_state.shape

company_state['DATE_1'].min(), company_state['DATE_1'].max()

company = company_state.sort_values('DATE_1', ascending=True)
company.isnull().sum()

company.head()

company["DATE_1"].value_counts()

company.dtypes

company = company_state.sort_values('DATE_1', ascending=True)
company.isnull().sum()

company.head(10)

company["DATE_1"].value_counts()

company = company.groupby('DATE_1')["VALUE_x"].sum().reset_index()
company.shape

company.head(10)

company = company.set_index("DATE_1")

company.head(10)

y = company['VALUE_x'].resample('M').sum()

print(y.shape)

y.isnull().sum()

y

y.shape

y.plot(figsize=(20,8))
plt.show()

train = y.loc[:'2017-08-31']
print(train.shape)
train

test = y.loc['2017-09-30':]
print(test.shape)
test

train.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
test.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
plt.show()

# Calculating Rolling Statistics

#determining rolling statistics (rolling mean at yearly level window=12)
rolmean=company.rolling(window=12).mean()


print(rolmean)


rolstd=company.rolling(window=12).std()
print(rolstd)

#plotting rolling statistics (Rolling Mean)
orig=plt.plot(y,color='blue',label="Original")
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
#std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean ")
plt.show(block=True)

#plotting rolling statistics(Rolling Deviation)
orig=plt.plot(y,color='blue',label="Original")
#mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Standard Deviation")
plt.show(block=False)

#ADFULLER test
from statsmodels.tsa.stattools import adfuller
print("ADfuller test")
dftest = adfuller(y,autolag ='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)

# # we reject null hypothesis : It is Stationary

#Simple ExpSmoothening
from statsmodels.tsa.api import SimpleExpSmoothing
Exp_Smooth = test.copy()
fit1 = SimpleExpSmoothing(train).fit(smoothing_level=0.01)
Exp_Smooth['SES'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Exp_Smooth['SES'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

Exp_Smooth.SES

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Exp_Smooth.SES))
print(rmse)

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()
plt.show()

# Double ExpSmoothening
from statsmodels.tsa.api import Holt
Holt_df = test.copy()
#soothing slope = beta
fit1 = Holt(train).fit(smoothing_level=0.1, smoothing_slope = 0.8)
Holt_df['Holt_linear'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_df['Holt_linear'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

Holt_df['Holt_linear']

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_df['Holt_linear']))
print(rmse)

#Triple ExpSmoothening
from statsmodels.tsa.api import ExponentialSmoothing
Holt_Winter_df = test.copy()
#soothing slope = beta
fit1 = ExponentialSmoothing(train,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Holt_Winter_df['Holt_Winter'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_Winter_df['Holt_Winter'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_Winter_df['Holt_Winter']))
print(rmse)

Holt_Winter_df['Holt_Winter']

from statsmodels.tsa.api import ExponentialSmoothing


fit1 = ExponentialSmoothing(y,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Y_predictions = fit1.forecast(steps=12)

y.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)

Y_predictions.plot(figsize=(10,6), title= 'Average Sales', fontsize=14) 
plt.show()

Y_predictions

train.plot(figsize=(20,8), title= 'Train - Average Sales ', fontsize=14,color="black",legend=True,label="Train Average Sales")
test.plot(figsize=(20,8), title='Test - Average Sales', fontsize=14,color="grey",legend=True,label="Test Average Sales")
Exp_Smooth['SES'].plot(figsize=(20,8),  fontsize=14,color="red",legend=True,label="Simple Exponential Smoothing")                                  
Holt_df['Holt_linear'].plot(figsize=(20,8),  fontsize=14,color="green",legend=True,label="Holt Linear")                                  
Holt_Winter_df['Holt_Winter'].plot(figsize=(20,8),  fontsize=14,color="blue",legend=True,label="Holt Winter") # seems Triple smooethening
Y_predictions.plot(figsize=(10,6),  fontsize=14,color="purple",legend=True,label="Predictions") 
plt.show() 
plt.title


# ARIMA MODEL

import pmdarima as pm

model = pm.auto_arima(y,start_p=0, max_p=3, d=None, max_d=2,start_q=0, max_q=3,  
                      start_P=0,max_P=3, D=None, max_D=2, start_Q=0, max_Q=3,
                      max_order=10, m=12, seasonal=True, information_criterion='aic',
                      test='adf',trace=True,random_state=10)

model

model.aic()

from statsmodels.tsa.statespace.sarimax import SARIMAX
model =SARIMAX(y, order=(0,1,1), seasonal_order=(0,0,2,12)).fit()

Y_pred=pred.predicted_mean
Y_test=y["2018-01-01":]

from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(Y_test,Y_pred))
print(rms)

#pred = model.get_prediction(start=pd.to_datetime('2017-01-31')) 
pred = model.get_forecast(steps=12)

plt.figure(figsize=(10,6))
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Validation Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

pred.predicted_mean

#pred.predicted_mean.to_excel("Timeseriesoutput.xlsx",index=True)



# State of Punjab

df.head(10)

df["STATE"].value_counts()

state =  df[df['STATE']=='Punjab']

state.shape

state

df1 =state[['DATE_1','VALUE_x']]

df1= df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1

df1['DATE_1'].min(), df1['DATE_1'].max()

df1.head()

df1["DATE_1"].value_counts()

df1.dtypes

df1 = df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1.head(10)

df1["DATE_1"].value_counts()

df1 = df1.groupby('DATE_1')["VALUE_x"].sum().reset_index()
df1.shape

df1.head(10)

df1 = df1.set_index("DATE_1")

df1.head(10)

df1.dtypes

y = df1['VALUE_x'].resample('M').sum()

y

print(y.shape)

y.isnull().sum()

y.plot(figsize=(20,8))
plt.show()

train = y.loc[:'2017-08-31']
print(train.shape)
train

test = y.loc['2017-09-30':]
print(test.shape)
test

train.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
test.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
plt.show()

#determining rolling statistics (rolling mean at yearly level window=12)
rolmean=company1.rolling(window=12).mean()


print(rolmean)

rolstd=company1.rolling(window=12).std()
print(rolstd)

#plotting rolling statistics (Rolling Mean)
orig=plt.plot(y,color='blue',label="Original")
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
#std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean ")
plt.show(block=True)

#plotting rolling statistics(Rolling Deviation)
orig=plt.plot(y,color='blue',label="Original")
#mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Standard Deviation")
plt.show(block=False)

#ADFULLER test
from statsmodels.tsa.stattools import adfuller
print("ADfuller test")
dftest = adfuller(y,autolag ='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)

# # we reject null hypothesis : It is Stationary

#Simple ExpSmoothening
from statsmodels.tsa.api import SimpleExpSmoothing
Exp_Smooth = test.copy()
fit1 = SimpleExpSmoothing(train).fit(smoothing_level=0.01)
Exp_Smooth['SES'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Exp_Smooth['SES'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

Exp_Smooth.SES

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Exp_Smooth.SES))
print(rmse)

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()
plt.show()

# Double ExpSmoothening
from statsmodels.tsa.api import Holt
Holt_df = test.copy()
#soothing slope = beta
fit1 = Holt(train).fit(smoothing_level=0.1, smoothing_slope = 0.8)
Holt_df['Holt_linear'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_df['Holt_linear'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

Holt_df['Holt_linear']

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_df['Holt_linear']))
print(rmse)

#Triple ExpSmoothening
from statsmodels.tsa.api import ExponentialSmoothing
Holt_Winter_df = test.copy()
#soothing slope = beta
fit1 = ExponentialSmoothing(train,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Holt_Winter_df['Holt_Winter'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_Winter_df['Holt_Winter'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_Winter_df['Holt_Winter']))
print(rmse)

Holt_Winter_df['Holt_Winter']

from statsmodels.tsa.api import ExponentialSmoothing


fit1 = ExponentialSmoothing(y,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Y_predictions = fit1.forecast(steps=12)

y.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)

Y_predictions.plot(figsize=(10,6), title= 'Average Sales', fontsize=14) 
plt.show()

Y_predictions

train.plot(figsize=(20,8), title= 'Train - Average Sales ', fontsize=14,color="black",legend=True,label="Train Average Sales")
test.plot(figsize=(20,8), title='Test - Average Sales', fontsize=14,color="grey",legend=True,label="Test Average Sales")
Exp_Smooth['SES'].plot(figsize=(20,8),  fontsize=14,color="red",legend=True,label="Simple Exponential Smoothing")                                  
Holt_df['Holt_linear'].plot(figsize=(20,8),  fontsize=14,color="green",legend=True,label="Holt Linear")                                  
Holt_Winter_df['Holt_Winter'].plot(figsize=(20,8),  fontsize=14,color="blue",legend=True,label="Holt Winter") # seems Triple smooethening
Y_predictions.plot(figsize=(10,6),  fontsize=14,color="purple",legend=True,label="Predictions") 
plt.show() 
plt.title


# ARIMA MODEL

import pmdarima as pm

model = pm.auto_arima(y,start_p=0, max_p=3, d=None, max_d=2,start_q=0, max_q=3,  
                      start_P=0,max_P=3, D=None, max_D=2, start_Q=0, max_Q=3,
                      max_order=10, m=12, seasonal=True, information_criterion='aic',
                      test='adf',trace=True,random_state=10)

model

model.aic()

from statsmodels.tsa.statespace.sarimax import SARIMAX
model =SARIMAX(y, order=(0,1,1), seasonal_order=(0,0,2,12)).fit()

Y_pred = pred.predicted_mean
Y_test = y['2018-01-01':]
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#pred = model.get_prediction(start=pd.to_datetime('2017-01-31')) 
pred = model.get_forecast(steps=12)

plt.figure(figsize=(10,6))
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Validation Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

pred.predicted_mean

#pred.predicted_mean.to_excel("Timeseriesoutput.xlsx",index=True)

# State of Haryana

df.head(10)

df["STATE"].value_counts()

state =  df[df['STATE']=='Haryana']

state.shape

state

df1 =state[['DATE_1','VALUE_x']]

df1= df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1

df1['DATE_1'].min(), df1['DATE_1'].max()

df1.head()

df1["DATE_1"].value_counts()

df1.dtypes

df1 = df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1.head(10)

df1 = df1.groupby('DATE_1')["VALUE_x"].sum().reset_index()
df1.shape

df1.head(10)

df1 = df1.set_index("DATE_1")

df1.head(10)

df1.dtypes

y = df1['VALUE_x'].resample('M').sum()

y

print(y.shape)

y.isnull().sum()

y.plot(figsize=(20,8))
plt.show()

train = y.loc[:'2017-08-31']
print(train.shape)
train

test = y.loc['2017-09-30':]
print(test.shape)
test

train.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
test.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
plt.show()

#determining rolling statistics (rolling mean at yearly level window=12)
rolmean=company1.rolling(window=12).mean()


print(rolmean)

rolstd=company1.rolling(window=12).std()
print(rolstd)

#plotting rolling statistics (Rolling Mean)
orig=plt.plot(y,color='blue',label="Original")
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
#std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean ")
plt.show(block=True)

#plotting rolling statistics(Rolling Deviation)
orig=plt.plot(y,color='blue',label="Original")
#mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Standard Deviation")
plt.show(block=False)

#ADFULLER test
from statsmodels.tsa.stattools import adfuller
print("ADfuller test")
dftest = adfuller(y,autolag ='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)

# # we reject null hypothesis : It is Stationary

#Simple ExpSmoothening
from statsmodels.tsa.api import SimpleExpSmoothing
Exp_Smooth = test.copy()
fit1 = SimpleExpSmoothing(train).fit(smoothing_level=0.01)
Exp_Smooth['SES'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Exp_Smooth['SES'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

Exp_Smooth.SES

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Exp_Smooth.SES))
print(rmse)

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()
plt.show()

# Double ExpSmoothening
from statsmodels.tsa.api import Holt
Holt_df = test.copy()
#soothing slope = beta
fit1 = Holt(train).fit(smoothing_level=0.1, smoothing_slope = 0.8)
Holt_df['Holt_linear'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_df['Holt_linear'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

Holt_df['Holt_linear']

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_df['Holt_linear']))
print(rmse)

#Triple ExpSmoothening
from statsmodels.tsa.api import ExponentialSmoothing
Holt_Winter_df = test.copy()
#soothing slope = beta
fit1 = ExponentialSmoothing(train,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Holt_Winter_df['Holt_Winter'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_Winter_df['Holt_Winter'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_Winter_df['Holt_Winter']))
print(rmse)

Holt_Winter_df['Holt_Winter']

from statsmodels.tsa.api import ExponentialSmoothing


fit1 = ExponentialSmoothing(y,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Y_predictions = fit1.forecast(steps=12)

y.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)

Y_predictions.plot(figsize=(10,6), title= 'Average Sales', fontsize=14) 
plt.show()

Y_predictions

train.plot(figsize=(20,8), title= 'Train - Average Sales ', fontsize=14,color="black",legend=True,label="Train Average Sales")
test.plot(figsize=(20,8), title='Test - Average Sales', fontsize=14,color="grey",legend=True,label="Test Average Sales")
Exp_Smooth['SES'].plot(figsize=(20,8),  fontsize=14,color="red",legend=True,label="Simple Exponential Smoothing")                                  
Holt_df['Holt_linear'].plot(figsize=(20,8),  fontsize=14,color="green",legend=True,label="Holt Linear")                                  
Holt_Winter_df['Holt_Winter'].plot(figsize=(20,8),  fontsize=14,color="blue",legend=True,label="Holt Winter") # seems Triple smooethening
Y_predictions.plot(figsize=(10,6),  fontsize=14,color="purple",legend=True,label="Predictions") 
plt.show() 
plt.title


# ARIMA MODEL

import pmdarima as pm

model = pm.auto_arima(y,start_p=0, max_p=3, d=None, max_d=2,start_q=0, max_q=3,  
                      start_P=0,max_P=3, D=None, max_D=2, start_Q=0, max_Q=3,
                      max_order=10, m=12, seasonal=True, information_criterion='aic',
                      test='adf',trace=True,random_state=10)

model

model.aic()

from statsmodels.tsa.statespace.sarimax import SARIMAX
model =SARIMAX(y, order=(0,1,1), seasonal_order=(0,0,2,12)).fit()

Y_pred = pred.predicted_mean
Y_test = y['2018-01-01':]
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#pred = model.get_prediction(start=pd.to_datetime('2017-01-31')) 
pred = model.get_forecast(steps=12)

plt.figure(figsize=(10,6))
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Validation Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

pred.predicted_mean

#pred.predicted_mean.to_excel("Timeseriesoutput.xlsx",index=True)

# STATE OF UTTARKHAND

df.tail(10)

df["STATE"].value_counts()

state =  df[df['STATE']=='Uttarakhand']

state.shape

state

df1 =state[['DATE_1','VALUE_x']]

df1= df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1

df1['DATE_1'].min(), df1['DATE_1'].max()

df1.head()

df1["DATE_1"].value_counts()

df1.dtypes

df1 = df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1.head(10)

df1 = df1.groupby('DATE_1')["VALUE_x"].sum().reset_index()
df1.shape

df1.head(10)

df1 = df1.set_index("DATE_1")

df1.head(10)

df1.dtypes

y = df1['VALUE_x'].resample('M').sum()

y

print(y.shape)

y.isnull().sum()

y.plot(figsize=(20,8))
plt.show()

train = y.loc[:'2017-08-31']
print(train.shape)
train

test = y.loc['2017-09-30':]
print(test.shape)
test

train.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
test.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
plt.show()

#determining rolling statistics (rolling mean at yearly level window=12)
rolmean=company1.rolling(window=12).mean()


print(rolmean)

rolstd=company1.rolling(window=12).std()
print(rolstd)

#plotting rolling statistics (Rolling Mean)
orig=plt.plot(y,color='blue',label="Original")
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
#std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean ")
plt.show(block=True)

#plotting rolling statistics(Rolling Deviation)
orig=plt.plot(y,color='blue',label="Original")
#mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Standard Deviation")
plt.show(block=False)

#ADFULLER test
from statsmodels.tsa.stattools import adfuller
print("ADfuller test")
dftest = adfuller(y,autolag ='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)

# # we reject null hypothesis : It is Stationary

#Simple ExpSmoothening
from statsmodels.tsa.api import SimpleExpSmoothing
Exp_Smooth = test.copy()
fit1 = SimpleExpSmoothing(train).fit(smoothing_level=0.01)
Exp_Smooth['SES'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Exp_Smooth['SES'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

Exp_Smooth.SES

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Exp_Smooth.SES))
print(rmse)

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()
plt.show()

# Double ExpSmoothening
from statsmodels.tsa.api import Holt
Holt_df = test.copy()
#soothing slope = beta
fit1 = Holt(train).fit(smoothing_level=0.1, smoothing_slope = 0.8)
Holt_df['Holt_linear'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_df['Holt_linear'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

Holt_df['Holt_linear']

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_df['Holt_linear']))
print(rmse)

#Triple ExpSmoothening
from statsmodels.tsa.api import ExponentialSmoothing
Holt_Winter_df = test.copy()
#soothing slope = beta
fit1 = ExponentialSmoothing(train,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Holt_Winter_df['Holt_Winter'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_Winter_df['Holt_Winter'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_Winter_df['Holt_Winter']))
print(rmse)

Holt_Winter_df['Holt_Winter']

from statsmodels.tsa.api import ExponentialSmoothing


fit1 = ExponentialSmoothing(y,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Y_predictions = fit1.forecast(steps=12)

y.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)

Y_predictions.plot(figsize=(10,6), title= 'Average Sales', fontsize=14) 
plt.show()

Y_predictions

train.plot(figsize=(20,8), title= 'Train - Average Sales ', fontsize=14,color="black",legend=True,label="Train Average Sales")
test.plot(figsize=(20,8), title='Test - Average Sales', fontsize=14,color="grey",legend=True,label="Test Average Sales")
Exp_Smooth['SES'].plot(figsize=(20,8),  fontsize=14,color="red",legend=True,label="Simple Exponential Smoothing")                                  
Holt_df['Holt_linear'].plot(figsize=(20,8),  fontsize=14,color="green",legend=True,label="Holt Linear")                                  
Holt_Winter_df['Holt_Winter'].plot(figsize=(20,8),  fontsize=14,color="blue",legend=True,label="Holt Winter") # seems Triple smooethening
Y_predictions.plot(figsize=(10,6),  fontsize=14,color="purple",legend=True,label="Predictions") 
plt.show() 
plt.title


# ARIMA MODEL

import pmdarima as pm

model = pm.auto_arima(y,start_p=0, max_p=3, d=None, max_d=2,start_q=0, max_q=3,  
                      start_P=0,max_P=3, D=None, max_D=2, start_Q=0, max_Q=3,
                      max_order=10, m=12, seasonal=True, information_criterion='aic',
                      test='adf',trace=True,random_state=10)

model

model.aic()

from statsmodels.tsa.statespace.sarimax import SARIMAX
model =SARIMAX(y, order=(0,1,1), seasonal_order=(0,0,2,12)).fit()

Y_pred = pred.predicted_mean
Y_test = y['2018-01-01':]
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#pred = model.get_prediction(start=pd.to_datetime('2017-01-31')) 
pred = model.get_forecast(steps=12)

plt.figure(figsize=(10,6))
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Validation Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

pred.predicted_mean

#pred.predicted_mean.to_excel("Timeseriesoutput.xlsx",index=True)

# STATE OF HIMACHAL PRADESH

df.head(10)

df["STATE"].value_counts()

state =  df[df['STATE']=='Himachal Pradesh']

state.shape

state

df1 =state[['DATE_1','VALUE_x']]

df1= df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1

df1['DATE_1'].min(), df1['DATE_1'].max()

df1.head()

df1["DATE_1"].value_counts()

df1.dtypes

df1 = df1.sort_values('DATE_1', ascending=True)
df1.isnull().sum()

df1.head(10)

df1 = df1.groupby('DATE_1')["VALUE_x"].sum().reset_index()
df1.shape

df1.head(10)

df1 = df1.set_index("DATE_1")

df1.head(10)

df1.dtypes

y = df1['VALUE_x'].resample('M').sum()

y

print(y.shape)

y.isnull().sum()

y.plot(figsize=(20,8))
plt.show()

train = y.loc[:'2017-08-31']
print(train.shape)
train

test = y.loc['2017-09-30':]
print(test.shape)
test

train.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
test.plot(figsize=(10,6), title = "Average Sales", fontsize = 14)
plt.show()

#determining rolling statistics (rolling mean at yearly level window=12)
rolmean=company1.rolling(window=12).mean()


print(rolmean)

rolstd=company1.rolling(window=12).std()
print(rolstd)

#plotting rolling statistics (Rolling Mean)
orig=plt.plot(y,color='blue',label="Original")
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
#std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean ")
plt.show(block=True)

#plotting rolling statistics(Rolling Deviation)
orig=plt.plot(y,color='blue',label="Original")
#mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,color='black',label="Rolling Std")
plt.legend(loc="best")
plt.title("Standard Deviation")
plt.show(block=False)

#ADFULLER test
from statsmodels.tsa.stattools import adfuller
print("ADfuller test")
dftest = adfuller(y,autolag ='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)

# # we reject null hypothesis : It is Stationary

#Simple ExpSmoothening
from statsmodels.tsa.api import SimpleExpSmoothing
Exp_Smooth = test.copy()
fit1 = SimpleExpSmoothing(train).fit(smoothing_level=0.01)
Exp_Smooth['SES'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Exp_Smooth['SES'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

Exp_Smooth.SES

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Exp_Smooth.SES))
print(rmse)

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()
plt.show()

# Double ExpSmoothening
from statsmodels.tsa.api import Holt
Holt_df = test.copy()
#soothing slope = beta
fit1 = Holt(train).fit(smoothing_level=0.1, smoothing_slope = 0.8)
Holt_df['Holt_linear'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_df['Holt_linear'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

Holt_df['Holt_linear']

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_df['Holt_linear']))
print(rmse)

#Triple ExpSmoothening
from statsmodels.tsa.api import ExponentialSmoothing
Holt_Winter_df = test.copy()
#soothing slope = beta
fit1 = ExponentialSmoothing(train,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Holt_Winter_df['Holt_Winter'] = fit1.forecast(steps=len(test))

train.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
test.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
Holt_Winter_df['Holt_Winter'].plot(figsize=(10,6), title= 'Average Sales', fontsize=14)
plt.show()

fit1.aic

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, Holt_Winter_df['Holt_Winter']))
print(rmse)

Holt_Winter_df['Holt_Winter']

from statsmodels.tsa.api import ExponentialSmoothing


fit1 = ExponentialSmoothing(y,seasonal_periods = 12, trend = 'add', seasonal='add').fit()
Y_predictions = fit1.forecast(steps=12)

y.plot(figsize=(10,6), title= 'Average Sales', fontsize=14)

Y_predictions.plot(figsize=(10,6), title= 'Average Sales', fontsize=14) 
plt.show()

Y_predictions

train.plot(figsize=(20,8), title= 'Train - Average Sales ', fontsize=14,color="black",legend=True,label="Train Average Sales")
test.plot(figsize=(20,8), title='Test - Average Sales', fontsize=14,color="grey",legend=True,label="Test Average Sales")
Exp_Smooth['SES'].plot(figsize=(20,8),  fontsize=14,color="red",legend=True,label="Simple Exponential Smoothing")                                  
Holt_df['Holt_linear'].plot(figsize=(20,8),  fontsize=14,color="green",legend=True,label="Holt Linear")                                  
Holt_Winter_df['Holt_Winter'].plot(figsize=(20,8),  fontsize=14,color="blue",legend=True,label="Holt Winter") # seems Triple smooethening
Y_predictions.plot(figsize=(10,6),  fontsize=14,color="purple",legend=True,label="Predictions") 
plt.show() 
plt.title


# ARIMA MODEL

import pmdarima as pm

model = pm.auto_arima(y,start_p=0, max_p=3, d=None, max_d=2,start_q=0, max_q=3,  
                      start_P=0,max_P=3, D=None, max_D=2, start_Q=0, max_Q=3,
                      max_order=10, m=12, seasonal=True, information_criterion='aic',
                      test='adf',trace=True,random_state=10)

model

model.aic()

from statsmodels.tsa.statespace.sarimax import SARIMAX
model =SARIMAX(y, order=(0,1,1), seasonal_order=(0,0,2,12)).fit()

Y_pred = pred.predicted_mean
Y_test = y['2018-01-01':]
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#pred = model.get_prediction(start=pd.to_datetime('2017-01-31')) 
pred = model.get_forecast(steps=12)

plt.figure(figsize=(10,6))
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Validation Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

pred.predicted_mean

#pred.predicted_mean.to_excel("Timeseriesoutput.xlsx",index=True)



