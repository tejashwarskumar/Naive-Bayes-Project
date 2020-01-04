import pandas as pd
salaryTrainData = pd.read_csv("C:/My Files/Excelr/11 - Naive Bayes/Assignment/SalaryData_Train.csv")
salaryTestData = pd.read_csv("C:/My Files/Excelr/11 - Naive Bayes/Assignment/SalaryData_Test.csv")

#salary Data is in categorical Format : change this to nominal fromat
from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"];

for i in columns:
    salaryTrainData[i] = prepocess.fit_transform(salaryTrainData[i])
    salaryTestData[i] = prepocess.fit_transform(salaryTestData[i])

columns_names = salaryTrainData.columns;
trainXData = salaryTrainData[columns_names[0:13]] 
trainYData = salaryTrainData[columns_names[13]] 
testXData = salaryTestData[columns_names[0:13]] 
tsetYData = salaryTestData[columns_names[13]] 

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

mnb = MultinomialNB();
gnb = GaussianNB();

mModel = mnb.fit(trainXData,trainYData)
testpredData = mModel.predict(testXData)
confusion_matrix(tsetYData,testpredData)
Accurancy_mnb = (10891+780)/(10891+469+2920+780)
Accurancy_mnb

gModel = gnb.fit(trainXData,trainYData)
testpredData_gnb = gModel.predict(testXData)
confusion_matrix(tsetYData,testpredData_gnb)
Accurancy_gnb = (10759+1209)/(10759+1209+2491+601)
Accurancy_gnb
