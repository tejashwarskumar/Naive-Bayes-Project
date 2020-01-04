library(readr)
library(e1071)

train_sal<-read.csv(file.choose())
View(train_sal)
test_sal<-read.csv(file.choose())
View(test_sal)

Model<-naiveBayes(train_sal$Salary~.,data=train_sal)
Model_pred<-predict(Model,test_sal)
CrossTable(Model_pred,test_sal$Salary,prop.chisq=FALSE,prop.t=FALSE,dnn=c("predicted","actual"))
mean(Model_pred==test_sal$Salary)