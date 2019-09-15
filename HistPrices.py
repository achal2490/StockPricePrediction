import matplotlib.pyplot as plt
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datetime as dt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


pd.options.display.max_columns = 50
data = pd.read_csv('DJIA_table.csv', encoding="ISO-8859-1")
data['Date'] = pd.to_datetime(data['Date'])
print(data.head(1))



# Split the training and test set
training_set, test_set = data[data['Date'] < '20150101'], data[data['Date'] > '20160530']
training_set = training_set.drop(['Date'], 1)
test_set = test_set.drop(['Date'], 1)

df = data
window_len = 10
# Create windows for training
LSTM_training_inputs = []

for i in range(len(training_set) - window_len):
    temp_set = training_set[i:(i + window_len)].copy()
    for col in list(temp_set):
       temp_set[col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values/ training_set['Close'][:-window_len].values) - 1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)


# Create windows for testing
LSTM_test_inputs = []
for i in range(len(test_set) - window_len):
    temp_set = test_set[i:(i + window_len)].copy()

    for col in list(temp_set):
       temp_set[col] = temp_set[col] / temp_set[col].iloc[0] - 1

    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values / test_set['Close'][:-window_len].values) - 1

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)


def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.10, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

# initialise model architecture
nn_model = build_model(LSTM_training_inputs, output_size=1, neurons = 32)
# model output is next price normalised to 10th previous closing price
# train model on data
# note: eth_history contains information on the training error per epoch
nn_history = nn_model.fit(LSTM_training_inputs, LSTM_training_outputs,epochs=5, batch_size=1, verbose=2, shuffle=True)


plt.title("Actual Vs Predicted Stock price")
plt.plot(LSTM_test_outputs, label = "actual")
plt.plot(nn_model.predict(LSTM_test_inputs), label = "predicted")
plt.legend()
#plt.show()


MAE = mean_absolute_error(LSTM_test_outputs, nn_model.predict(LSTM_test_inputs))
RMSE = math.sqrt(mean_squared_error(LSTM_test_outputs, nn_model.predict(LSTM_test_inputs)))
print('The Mean Absolute Error is: {}'.format(MAE))
print('The Mean Squared Error is: {}'.format(RMSE))

# save the model to disk
from sklearn.externals import joblib
joblib.dump(nn_model, 'LSTM.sav')

print(LSTM_test_outputs)
print(nn_model.predict(LSTM_test_inputs))