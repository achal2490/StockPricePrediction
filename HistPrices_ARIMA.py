import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


pd.options.display.max_columns = 50
data = pd.read_csv('DJIA_table.csv', encoding="ISO-8859-1")
data['Date'] = pd.to_datetime(data['Date'])
print(data.head(1))

df = data
window_len = 10



# Split the training and test set
training_set, test_set = data[data['Date'] < '20150101'], data[data['Date'] > '20141231']
training_set = training_set.drop(['Date'], 1)
test_set = test_set.drop(['Date'], 1)


train_ar = training_set['Close']
test_ar = test_set['Close']
history = [x for x in train_ar]
y = test_ar


# make first prediction
predictions = list()
# rolling forecasts
for i in range(len(y)):
    # predict
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
# report performance
mae = mean_absolute_error(y, predictions)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: '+str(rmse))


print(predictions)
print(test_ar)