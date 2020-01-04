import pandas as pd
import numpy as nm
spamData = pd.read_csv("C:/My Files/Excelr/11 - Naive Bayes/Assignment/sms_raw_NB.csv",encoding = "ISO-8859-1")

import re
stop_words=[];
with open('C:/My Files/Excelr/11 - Naive Bayes/Assignment/stop.txt') as f:stop_words = f.read()
stop_words = stop_words.split("\n")
spamdataCon = ''.join(spamData.text)
spamdataCon = re.sub("[^A-Za-z" "]+"," ",spamdataCon).lower()
spamdataCon = re.sub("[0-9" "]+"," ",spamdataCon).lower()

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

spamData.text = spamData.text.apply(cleaning_text)
spamData = spamData.loc[spamData.text != " ",:]

def split_into_words(i):
    return [word for word in i.split(" ")]

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

spamTrain,spamTest = train_test_split(spamData,test_size=0.3)
spamData_con_fn = CountVectorizer(analyzer=split_into_words).fit(spamData.text)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

spamData_con = spamData_con_fn.transform(spamData.text)
spamTrain_con = spamData_con_fn.transform(spamTrain.text)
spamTest_con = spamData_con_fn.transform(spamTest.text)
spamTrain_con.shape

mnb = MultinomialNB()
mModel = mnb.fit(spamTrain_con,spamTrain.type)
mnb_pred=mModel.predict(spamTrain_con)
accurancy_mnb = nm.mean(mnb_pred == spamTrain.type)
accurancy_mnb

gnb = GaussianNB()
gModel = gnb.fit(spamTrain_con.toarray(),spamTrain.type.values)
gnb_pred = gModel.predict(spamTrain_con.toarray())
accurancy_gnb = nm.mean(gnb_pred == spamTrain.type)
accurancy_gnb

tfidf_spamdata = TfidfTransformer().fit(spamData_con)
tfidf_spamTrain_con = tfidf_spamdata.transform(spamTrain_con)
tfidf_spamTest_con = tfidf_spamdata.transform().fit(spamTest_con)

mnb_tfidf = MultinomialNB()
mModel_tfidf = mnb_tfidf.fit(tfidf_spamTrain_con,spamTrain.type)
mnb_pred_tfidf=mModel_tfidf.predict(spamTrain_con)
accurancy_mnb_tfidf = nm.mean(mnb_pred_tfidf == spamTrain.type)
accurancy_mnb_tfidf

gnb_tfidf = GaussianNB()
gModel_tfidf = gnb_tfidf.fit(tfidf_spamTrain_con.toarray(),spamTrain.type.values)
gnb_pred_tfidf = gModel_tfidf.predict(tfidf_spamTrain_con.toarray())
accurancy_gnb_tfidf = nm.mean(gnb_pred_tfidf == spamTrain.type)
accurancy_gnb_tfidf

#Test on Train Data
test_predict = mModel.predict(spamTest_con)
accurancy_test = nm.mean(test_predict == spamTest.type)
accurancy_test