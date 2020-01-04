library(readr)
sms<-read.csv(file.choose(), stringsAsFactors <- FALSE)
str(sms)
round(prop.table(table(sms$type))*100, digits <- 1)
sms$type <- factor(sms$type)

library(tm)
sms_corpus <- VCorpus(VectorSource(sms$text))
print(sms_corpus)
inspect(sms_corpus[1:3])

corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean[1:3])

corpus_clean <- tm_map(corpus_clean, PlainTextDocument)
dtm <- DocumentTermMatrix(corpus_clean)
str(dtm)

sms.train <- sms[1:4200, ]
sms.test  <- sms[4201:5559,]
dtm.train <- dtm[1:4200, ]
dtm.test  <- dtm[4201:5559,]
corpus.train <- corpus_clean[1:4200]
corpus.test  <- corpus_clean[4201:5559]

round(prop.table(table(sms.train$type))*100)
round(prop.table(table(sms.test$type))*100)

library(wordcloud)
wordcloud(corpus.train,min.freq=40,random.order = FALSE)
spam <- subset(sms.train, type == "spam")
ham  <- subset(sms.train, type == "ham")
wordcloud(spam$text,max.words=40,scale=c(3, 0, 5))
wordcloud(ham$text,max.words=40,scale=c(3, 0, 5))

freq_terms <- findFreqTerms(dtm.train, 5)
reduced_dtm.train <- DocumentTermMatrix(corpus.train, list(dictionary<-freq_terms))
reduced_dtm.test <-  DocumentTermMatrix(corpus.test, list(dictionary<-freq_terms))
ncol(reduced_dtm.train)
ncol(reduced_dtm.test)
convert_counts <- function(x)
{
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels <- c(0, 1), labels<-c("No", "Yes"))
  return (x)
}
reduced_dtm.train <- apply(reduced_dtm.train, MARGIN<-2, convert_counts)
reduced_dtm.test  <- apply(reduced_dtm.test, MARGIN<-2, convert_counts)

library(e1071)
library(gmodels)
sms_classifier <- naiveBayes(reduced_dtm.train, sms.train$type)
sms_test.predicted <- predict(sms_classifier,reduced_dtm.test)
CrossTable(sms_test.predicted,sms.test$type,prop.chisq = FALSE,prop.t = FALSE,dnn= c("predicted", "actual"))
mean(sms_test.predicted==sms.test$type)
sms_classifier2 <- naiveBayes(reduced_dtm.train,sms.train$type,laplace = 1)
sms_test.predicted2 <- predict(sms_classifier2,reduced_dtm.test)
CrossTable(sms_test.predicted2,sms.test$type,prop.chisq = FALSE,prop.t= FALSE,dnn= c("predicted", "actual"))
mean(sms_test.predicted2==sms.test$type)
