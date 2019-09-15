import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.svm import SVC

# Read in the data
#Data = pd.read_csv('Full_Data.csv', encoding = "ISO-8859-1")
#Data.head(1)
data = pd.read_csv('Combined_News_DJIA.csv', encoding = "ISO-8859-1")
print(data.head(1))

train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']


# Removing punctuations
slicedData= train.iloc[:,2:27]
slicedData.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
slicedData.columns= new_Index
print(slicedData.head(5))

# Convertng headlines to lower case
for index in new_Index:
    slicedData[index]=slicedData[index].str.lower()
print(slicedData.head(1))

#Concatenate
headlines = []
for row in range(0,len(slicedData.index)):
    headlines.append(' '.join(str(x) for x in slicedData.iloc[row,0:25]))

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

print(headlines[0])

#Basic Tokenaization 1 gram
basicvectorizer = CountVectorizer(ngram_range=(1,1))
basictrain = basicvectorizer.fit_transform(headlines)
basictest = basicvectorizer.transform(testheadlines)

basicmodel = svm.SVC(C=1, class_weight='balanced',kernel='rbf', gamma=0.100000000000000000000001, tol=1e-10)
basicmodel = basicmodel.fit(basictrain, train["Label"])

predictions = basicmodel.predict(basictest.toarray())

print(pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"]))

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print (classification_report(test["Label"], predictions))
print (accuracy_score(test["Label"], predictions))

# 2- Gram Model
basicvectorizer2 = CountVectorizer(ngram_range=(1,2))
basictrain2 = basicvectorizer2.fit_transform(headlines)
basictest2 = basicvectorizer2.transform(testheadlines)

basicmodel2 = svm.SVC(C=1, class_weight='balanced',kernel='rbf', gamma=0.100000000000000000000001, tol=1e-10)
basicmodel2 = basicmodel2.fit(basictrain2, train["Label"])

predictions2 = basicmodel2.predict(basictest2)
print(pd.crosstab(test["Label"], predictions2, rownames=["Actual"], colnames=["Predicted"]))
print (classification_report(test["Label"], predictions2))
print (accuracy_score(test["Label"], predictions2))

# 3- Gram Model
basicvectorizer3 = CountVectorizer(ngram_range=(2,3))
basictrain3 = basicvectorizer3.fit_transform(headlines)
basictest3 = basicvectorizer3.transform(testheadlines)

basicmodel3 = svm.SVC(C=1, class_weight='balanced',kernel='rbf', gamma=0.100000000000000000000001, tol=1e-10)
basicmodel3 = basicmodel3.fit(basictrain3, train["Label"])

predictions3 = basicmodel3.predict(basictest3)
print(pd.crosstab(test["Label"], predictions3, rownames=["Actual"], colnames=["Predicted"]))
print (classification_report(test["Label"], predictions3))
print (accuracy_score(test["Label"], predictions3))

# save the model to disk
from sklearn.externals import joblib
joblib.dump(basicmodel, 'NLSVM_NGram1.sav')
joblib.dump(basicmodel2, 'NLSVM_NGram2.sav')
joblib.dump(basicmodel3, 'NLSVM_NGram3.sav')



print(test.tail(5))
print(predictions)
print(predictions2)
print(predictions3)