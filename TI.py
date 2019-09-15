import numpy as np
import pandas as pd
import os
import random
import copy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

# Read in the data
pd.options.display.max_columns = 50
data = pd.read_csv('DJIA_table.csv', encoding="ISO-8859-1")
data['Date'] = pd.to_datetime(data['Date'])
print(data.head(1))
TechIndicator = copy.deepcopy(data)


# Relative Strength Index
# Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
# Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};
#        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};

def rsi(values):
    up = values[values > 0].mean()
    down = -1 * values[values < 0].mean()
    return 100 * up / (up + down)


TechIndicator['Momentum_1D'] = (TechIndicator['Close'] - TechIndicator['Close'].shift(1)).fillna(0)
TechIndicator['RSI_14D'] = TechIndicator['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)

print(TechIndicator.tail(3))


# Bollinger Bands
def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    ave = price.rolling(window=length, center=False).mean()
    sd = price.rolling(window=length, center=False).std()
    upband = ave + (sd * numsd)
    dnband = ave - (sd * numsd)
    return np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


TechIndicator['BB_Middle_Band'], TechIndicator['BB_Upper_Band'], TechIndicator[
    'BB_Lower_Band'] = bbands(TechIndicator['Close'], length=20, numsd=1)
TechIndicator['BB_Middle_Band'] = TechIndicator['BB_Middle_Band'].fillna(0)
TechIndicator['BB_Upper_Band'] = TechIndicator['BB_Upper_Band'].fillna(0)
TechIndicator['BB_Lower_Band'] = TechIndicator['BB_Lower_Band'].fillna(0)
print(TechIndicator.tail(3))



# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
TechIndicator['ROC'] = ((TechIndicator['Close'] - TechIndicator['Close'].shift(12))/(TechIndicator['Close'].shift(12)))*100
TechIndicator = TechIndicator.fillna(0)
print(TechIndicator.tail(3))


# MACD
TechIndicator['26_ema'] = TechIndicator['Close'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()
TechIndicator['12_ema'] = TechIndicator['Close'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()
TechIndicator['MACD'] = TechIndicator['12_ema'] - TechIndicator['26_ema']
TechIndicator['Signal'] = TechIndicator['MACD'].ewm(span=9,min_periods=0,adjust=True,ignore_na=False).mean()
TechIndicator = TechIndicator.fillna(0)
print(TechIndicator.tail(3))


# Adaptive Moving Average
def KAMA(price, n=10, pow1=2, pow2=30):
    ''' kama indicator '''
    ''' accepts pandas dataframe of prices '''

    absDiffx = abs(price - price.shift(1) )

    ER_num = abs( price - price.shift(n) )
    ER_den = absDiffx.rolling(window=n,center=False).sum()
    ER = ER_num / ER_den

    sc = ( ER*(2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0) ) ** 2.0


    answer = np.zeros(sc.size)
    N = len(answer)
    first_value = True

    for i in range(N):
        if sc[i] != sc[i]:
            answer[i] = np.nan
        else:
            if first_value:
                answer[i] = price[i]
                first_value = False
            else:
                answer[i] = answer[i-1] + sc[i] * (price[i] - answer[i-1])
    return answer

TechIndicator['KAMA'] = KAMA(TechIndicator['Close'])
TechIndicator = TechIndicator.fillna(0)
print(TechIndicator.tail(3))

# Chaikin Oscilator
def CMFlow(df, tf):
    CHMF = []
    MFMs = []
    MFVs = []
    x = tf

    while x < len(df['Date']):
        PeriodVolume = 0
        volRange = df['Volume'][x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol

        MFM = ((df['Close'][x] - df['Low'][x]) - (df['High'][x] - df['Close'][x])) / (df['High'][x] - df['Low'][x])
        MFV = MFM * PeriodVolume

        MFMs.append(MFM)
        MFVs.append(MFV)
        x += 1

    y = tf
    while y < len(MFVs):
        PeriodVolume = 0
        volRange = df['Volume'][x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol
        consider = MFVs[y - tf:y]
        tfsMFV = 0

        for eachMFV in consider:
            tfsMFV += eachMFV

        tfsCMF = tfsMFV / PeriodVolume
        CHMF.append(tfsCMF)
        y += 1
    return CHMF

listofzeros = [0] * 40
CHMF = CMFlow(TechIndicator, 20)
if len(CHMF)==0:
    CHMF = [0] * TechIndicator.shape[0]
    TechIndicator['Chaikin_MF'] = CHMF
else:
    TechIndicator['Chaikin_MF'] = listofzeros+CHMF
print(TechIndicator.tail(3))


######Plotting only for prediction years##########
TrainTechnicalIndicator = TechIndicator[TechIndicator['Date'] < '20150101']
TechIndicator = TechIndicator[TechIndicator['Date'] > '20141231']
TechIndicator = TechIndicator[TechIndicator['Date'] < '20160530']
TechIndicator.index = TechIndicator['Date']
TechIndicator = TechIndicator.drop(labels = ['Date'], axis = 1)


fig = plt.figure()
for j in range(6):
    ax = plt.subplot(3, 2, j + 1)
    if j == 0:
        plt.plot(TechIndicator.index, TechIndicator['RSI_14D'],label="RSI")
        plt.xticks(rotation=30)
    elif j == 1:
        plt.fill_between(TechIndicator.index, TechIndicator['BB_Upper_Band'], TechIndicator['BB_Lower_Band'],
                         color='grey', label="Band Range")
        # Plot Adjust Closing Price and Moving Averages
        plt.plot(TechIndicator.index, TechIndicator['Close'], color='red', lw=2, label="Close")
        plt.plot(TechIndicator.index, TechIndicator['BB_Middle_Band'], color='black', lw=2, label="Middle Band")
        plt.xticks(rotation=30)
    elif j == 2:
        plt.plot(TechIndicator.index, TechIndicator['ROC'], 'red', label="ROC")
        plt.xticks(rotation=30)
    elif j == 3:
        plt.plot(TechIndicator.index, TechIndicator['KAMA'], 'blue', label="AMA")
        plt.plot(TechIndicator.index, TechIndicator['Close'], 'red', label="Close", alpha=0.5)
        plt.xticks(rotation=30)
    elif j == 5:
        t = ax.fill(TechIndicator.index, TechIndicator['Chaikin_MF'], 'b', label="Chaikin MF")
        t = ax.set_ylabel(" Chaikin Money Flow")
        plt.xticks(rotation=30)
    elif j == 4:
        plt.plot(TechIndicator.index, TechIndicator['MACD'], 'green', label="MACD")
        plt.axhline(0, color='b', ls='dashed', alpha=0.3)
        plt.plot(TechIndicator.index, TechIndicator['Signal'], 'red', label="Signal", alpha=0.7)
        t = ax.set_ylabel("MACD")
        plt.xticks(rotation=30)
    ax.legend(loc='best')
    ax.set_xlabel("Date")
fig.tight_layout()
plt.show()
plt.close()

trainX = TrainTechnicalIndicator[['Momentum_1D','RSI_14D','BB_Middle_Band','BB_Upper_Band','BB_Lower_Band','ROC','26_ema','12_ema','MACD','KAMA','Chaikin_MF']]
trainY = TrainTechnicalIndicator[['Close']]

testX = TechIndicator[['Momentum_1D','RSI_14D','BB_Middle_Band','BB_Upper_Band','BB_Lower_Band','ROC','26_ema','12_ema','MACD','KAMA','Chaikin_MF']]
testY = TechIndicator[['Close']]

# with sklearn
regr = linear_model.LinearRegression()
mod = regr.fit(trainX, trainY)

from sklearn.metrics import r2_score
print(r2_score(testY, mod.predict(testX)))

import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mae = mean_absolute_error(testY, mod.predict(testX))
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(testY, mod.predict(testX)))
print('RMSE: '+str(rmse))


print(testY.head(5))

print(mod.predict(testX)[1:6])

