library(caTools)

set.seed(456)

pdata = read.csv("/home/long/TTU-SOURCES/insured-prediction/data/pdata.csv", header = TRUE)

pdata = pdata[sample(nrow(pdata), 1000), ]
nrow(pdata)
# create split with 70% is TRUE (this will be used as training set)
spl = sample.split(pdata$Overall_Auth_Status, SplitRatio = 0.7)
trainSparse = subset(pdata, spl == TRUE)
testSparse = subset(pdata, spl == FALSE)



## trainSparse now has 700 rows (70%) 
nrow(trainSparse)
str(testSparse)
## testSparse now has 300 rows (30%)
nrow(testSparse)
# CART Model
library(rpart)
library(rpart.plot)
tweetCART <- rpart(Overall_Auth_Status ~ feed_id_full_path + Diagnosis_Name + Provider + SibSp, data=trainSparse, method="class")
prp(tweetCART)


# Predict using the trainig set. Because the CART tree assigns the same predicted probability to each leaf node and there are a small number of leaf nodes compared to data points, we expect exactly the same maximum predicted probability.
predictCart <- predict(tweetCART, newdata=testSparse, type="class")
summary(predictCart)

str(predictCart)
str(testSparse$Overall_Auth_Status)

## accuracy test
confusionMatrix = table(testSparse$Overall_Auth_Status, predictCart)
confusionMatrix
confusionMatrix[1,1]

a = confusionMatrix[2,2] 
b = confusionMatrix[2,1]
c = confusionMatrix[1, 2]
d = confusionMatrix[1, 1]
precision = a / (a + c)
precision

recall = a / (a + b)
recall

fMeasure = 2*a / (2*a + b + c)
fMeasure

accuracy = (a + d) / (a + b + c +d)
accuracy

message(paste("accuracy: ", accuracy, "; precision: ", precision, "; recall: ", recall, "; f-measure: ", fMeasure))